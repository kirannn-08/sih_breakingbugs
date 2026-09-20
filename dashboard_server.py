"""
Web Dashboard Server for Decentralized AMR Fleet Coordination.
Smart India Hackathon Problem Statement 26123.

Provides a live, high-framerate WebSocket & REST service bridging the
headless Python simulation, comms mediator, and task allocation to
an interactive ground-station web frontend.

Architectural Compliance:
- Server acts as an OBSERVER of fleet telemetry and BROADCASTER of tasks.
- It never commands robot motion or path planning (preserving the core claim).
- Comms fault injection (link cutting, packet loss) operates directly on the
  isolation boundary (CommsMediator).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any

import tornado.ioloop
import tornado.web
import tornado.websocket

from amr_msgs import (BROADCAST, INF_COST, Bid, Claim, Header, Intent, MsgType,
                      Release, RobotMode, RobotState, Task, make_header, wrap)
from comms import CommsMediator, DeadZone
from coordination import DeadlockDetector, adapt_speed, resolve_auction
from learned import OBS_DIM, PolicyNet
from planner import path_to_reservations
from sim2d import DT, Simulation
from sim_manager import SimulationManager, dispatch
from warehouse_map import CELL_SIZE, H, W, WarehouseMap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AMRDashboard")


# ==============================================================================
# TORNADO HTTP & WEBSOCKET HANDLERS
# ==============================================================================

MANAGER = SimulationManager()
CONNECTED_SOCKETS: set[tornado.websocket.WebSocketHandler] = set()


class SimWebSocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin: str) -> bool:
        return True  # Allow local network / dashboard connections

    def open(self) -> None:
        CONNECTED_SOCKETS.add(self)
        logger.info(f"WebSocket client connected ({len(CONNECTED_SOCKETS)} total)")
        # Send initial full map definition and state
        self.write_message(json.dumps({
            "type": "init",
            "map": MANAGER.get_map_data(),
            "state": MANAGER.get_state()
        }))

    def on_close(self) -> None:
        CONNECTED_SOCKETS.discard(self)
        logger.info(f"WebSocket client disconnected ({len(CONNECTED_SOCKETS)} remaining)")

    def on_message(self, message: str) -> None:
        try:
            cmd = json.loads(message)
            res = dispatch_action(cmd)
            if res.get("type") == "scenario_result":
                self.write_message(json.dumps(res))
            broadcast_state()
        except Exception as e:
            logger.error(f"Error handling websocket message: {e}")
            self.write_message(json.dumps({"type": "error", "error": str(e)}))


def dispatch_action(cmd: dict[str, Any]) -> dict[str, Any]:
    """Thin wrapper: the dispatcher itself lives in sim_manager so the
    browser build runs the identical command vocabulary."""
    return dispatch(MANAGER, cmd)


def broadcast_state() -> None:
    if not CONNECTED_SOCKETS:
        return
    msg = json.dumps({"type": "state", "state": MANAGER.get_state()})
    for ws in list(CONNECTED_SOCKETS):
        try:
            ws.write_message(msg)
        except Exception:
            CONNECTED_SOCKETS.discard(ws)


class MapHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(MANAGER.get_map_data()))


class StateHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(MANAGER.get_state()))


class BenchmarkHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        self.set_header("Content-Type", "application/json")
        benchmark_file = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
        if os.path.exists(benchmark_file):
            with open(benchmark_file, "r") as f:
                data = json.load(f)
            self.write(json.dumps(data))
        else:
            self.write(json.dumps({"error": "No benchmark results file found"}))


class ActionHandler(tornado.web.RequestHandler):
    """REST mirror of the WebSocket command set -- same dispatcher, no drift."""

    def post(self) -> None:
        try:
            res = dispatch_action(json.loads(self.request.body))
            self.write(json.dumps(res))
            broadcast_state()
        except ValueError as e:
            self.set_status(400)
            self.write(json.dumps({"error": str(e)}))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


class CommsDemoHandler(tornado.web.RequestHandler):
    """Serves the comms workstream's standalone page, and only that file.

    A StaticFileHandler pointed at communication_protocols/ would also hand out
    server.py and its siblings as downloadable source once this is hosted, so
    the route is scoped to the single page the topbar button links to.
    """

    PAGE = os.path.join(os.path.dirname(__file__),
                        "communication_protocols", "amr_simulation.html")

    def get(self) -> None:
        if not os.path.exists(self.PAGE):
            self.set_status(404)
            self.write("comms demo page not found")
            return
        self.set_header("Content-Type", "text/html; charset=utf-8")
        with open(self.PAGE, "r", encoding="utf-8") as f:
            self.write(f.read())


def sim_tick() -> None:
    """Periodic tick driving the simulation at target frame rate."""
    if MANAGER.is_running:
        # Step the simulation according to speed multiplier
        steps = max(1, int(MANAGER.speed_multiplier))
        for _ in range(steps):
            MANAGER.step()
        broadcast_state()


def make_app(static_path: str) -> tornado.web.Application:
    return tornado.web.Application([
        (r"/ws", SimWebSocketHandler),
        (r"/api/map", MapHandler),
        (r"/api/state", StateHandler),
        (r"/api/benchmark", BenchmarkHandler),
        (r"/api/action", ActionHandler),
        (r"/comms/?", CommsDemoHandler),
        (r"/(.*)", tornado.web.StaticFileHandler, {
            "path": static_path,
            "default_filename": "index.html"
        }),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="AMR Fleet Coordination Mission Control Server")
    # Render/Railway/Fly assign the port at runtime and health-check it; binding
    # a hardcoded 8080 there fails the deploy. The flag still wins locally.
    default_port = int(os.environ.get("PORT", 8080))
    parser.add_argument("--port", type=int, default=default_port,
                        help=f"HTTP/WebSocket port (default: {default_port})")
    parser.add_argument("--robots", type=int, default=4, help="Number of robots (default: 4)")
    parser.add_argument("--seed", type=int, default=1, help="Simulation random seed (default: 1)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address")
    args = parser.parse_args()

    MANAGER.n_robots = args.robots
    MANAGER.seed = args.seed
    MANAGER.reset()

    web_dir = os.path.join(os.path.dirname(__file__), "web")
    os.makedirs(web_dir, exist_ok=True)

    app = make_app(web_dir)
    app.listen(args.port, address=args.host)

    # 100ms interval = 10Hz base step rate (matching DT = 0.1s in sim2d.py)
    tick_loop = tornado.ioloop.PeriodicCallback(sim_tick, 100)
    tick_loop.start()

    logger.info(f"============================================================")
    logger.info(f"🚀 AMR Fleet Coordination Mission Control Server Running!")
    logger.info(f"   URL: http://localhost:{args.port}/")
    logger.info(f"   WebSocket: ws://localhost:{args.port}/ws")
    logger.info(f"   Robots: {args.robots} AMRs | Seed: {args.seed}")
    logger.info(f"============================================================")

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
