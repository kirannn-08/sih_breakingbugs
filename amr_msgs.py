"""
AMR-Link message schemas.  ===  FROZEN CONTRACT  ===

Every team member builds against THIS FILE. Do not change a field without a
team-wide decision, because four workstreams depend on it:

    dashboard   -> reads Telemetry
    comms       -> routes LinkPacket
    algorithms  -> produces/consumes everything else
    sim         -> generates RobotState

Transport-agnostic on purpose. JSON over whatever you pick (ROS 2 topic,
UDP socket, TCP). Swap to msgpack later if bandwidth matters; the schema
does not change.

Conventions
-----------
  * positions in METRES, world frame, float
  * grid cells in (cx, cy) ints
  * angles in RADIANS, -pi..pi
  * times in SECONDS, float, sim-clock
  * robot_id: int 1..N  (0 is reserved = "none"/broadcast)
  * lower cost wins, lower priority tuple = HIGHER priority
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.1.0"

BROADCAST = 0  # robot_id 0 means "to everyone" / "no robot"


# --------------------------------------------------------------------------
# Enums  (wire format is the string value, so logs stay readable)
# --------------------------------------------------------------------------

class RobotMode(str, Enum):
    IDLE = "IDLE"
    BIDDING = "BIDDING"
    TO_PICKUP = "TO_PICKUP"
    LOADED = "LOADED"
    TO_DROPOFF = "TO_DROPOFF"
    CHARGING = "CHARGING"
    BLOCKED = "BLOCKED"
    DEGRADED = "DEGRADED"      # comms lost, running on local sensing only
    FAULT = "FAULT"


class MsgType(str, Enum):
    ROBOT_STATE = "ROBOT_STATE"
    INTENT = "INTENT"
    TASK = "TASK"
    BID = "BID"
    CLAIM = "CLAIM"
    RELEASE = "RELEASE"
    OBSTACLE = "OBSTACLE"
    REROUTE = "REROUTE"
    COORDINATION = "COORDINATION"
    WAIT_FOR = "WAIT_FOR"
    TELEMETRY = "TELEMETRY"


class ReleaseReason(str, Enum):
    BATTERY = "BATTERY"
    BLOCKED = "BLOCKED"
    FAULT = "FAULT"
    PREEMPTED = "PREEMPTED"


class RerouteReason(str, Enum):
    BLOCKAGE = "BLOCKAGE"
    CONGESTION = "CONGESTION"
    CONFLICT = "CONFLICT"
    DEADLOCK = "DEADLOCK"


class Resolution(str, Enum):
    I_YIELD = "I_YIELD"
    I_PROCEED = "I_PROCEED"
    I_SLOW = "I_SLOW"


class ObstacleKind(str, Enum):
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------
# Header  (on every message)
# --------------------------------------------------------------------------

@dataclass
class Header:
    robot_id: int                 # sender
    seq: int                      # monotonic per sender; gaps = packet loss
    stamp: float                  # sender clock, seconds
    ttl_ms: int = 2000            # after this, receivers treat payload as stale

    def age(self, now: float) -> float:
        return now - self.stamp

    def is_stale(self, now: float) -> bool:
        return self.age(now) * 1000.0 > self.ttl_ms


# --------------------------------------------------------------------------
# Core state + intent
# --------------------------------------------------------------------------

@dataclass
class RobotState:
    """Broadcast at 5-10 Hz. The 'where am I' message."""
    header: Header
    x: float
    y: float
    theta: float
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    mode: str = RobotMode.IDLE.value
    battery_soc: float = 1.0          # 0..1
    payload_kg: float = 0.0
    capacity_kg: float = 20.0
    current_task: int = 0             # 0 = none
    lidar_ok: bool = True
    comms_ok: bool = True
    motors_ok: bool = True


@dataclass
class Reservation:
    """One space-time cell claim. The core coordination primitive."""
    cx: int
    cy: int
    t_enter: float
    t_exit: float

    def overlaps(self, other: "Reservation", tau: float = 1.5) -> bool:
        if (self.cx, self.cy) != (other.cx, other.cy):
            return False
        return not (self.t_exit + tau < other.t_enter
                    or other.t_exit + tau < self.t_enter)


@dataclass
class Intent:
    """
    Broadcast on change + 2 Hz heartbeat.

    This is the message that makes the whole system work. Peers use
    `reservations` to predict conflicts BEFORE they happen, instead of
    reacting once robots are already nose to nose.
    """
    header: Header
    task_id: int = 0
    path_cells: list[list[int]] = field(default_factory=list)   # [[cx,cy],...]
    reservations: list[Reservation] = field(default_factory=list)
    next_wx: float = 0.0
    next_wy: float = 0.0
    eta_goal: float = 0.0
    nominal_speed: float = 0.0
    priority: list[float] = field(default_factory=list)   # see priority_tuple()


# --------------------------------------------------------------------------
# Task allocation
# --------------------------------------------------------------------------

@dataclass
class Task:
    """Server -> all robots. BROADCAST ONLY. Never names a robot."""
    header: Header
    task_id: int
    pickup_cx: int
    pickup_cy: int
    dropoff_cx: int
    dropoff_cy: int
    payload_kg: float = 1.0
    priority_class: int = 1           # lower = more urgent
    deadline: float = 0.0             # 0 = none
    announced_at: float = 0.0


@dataclass
class Bid:
    header: Header
    task_id: int
    cost: float                       # lower wins; use INF_COST if infeasible
    feasible: bool = True
    eta: float = 0.0


INF_COST = 1e12


@dataclass
class Claim:
    header: Header
    task_id: int
    winning_cost: float


@dataclass
class Release:
    header: Header
    task_id: int
    reason: str = ReleaseReason.BLOCKED.value


# --------------------------------------------------------------------------
# Environment + coordination
# --------------------------------------------------------------------------

@dataclass
class Obstacle:
    header: Header
    cells: list[list[int]] = field(default_factory=list)
    kind: str = ObstacleKind.UNKNOWN.value
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 1.0
    observed_at: float = 0.0


@dataclass
class Reroute:
    header: Header
    task_id: int
    blocked_cells: list[list[int]] = field(default_factory=list)
    new_path: list[list[int]] = field(default_factory=list)
    new_reservations: list[Reservation] = field(default_factory=list)
    reason: str = RerouteReason.BLOCKAGE.value


@dataclass
class Coordination:
    """Speed negotiation. The anti-stop-and-wait message."""
    header: Header
    conflict_with: int
    conflict_cx: int
    conflict_cy: int
    conflict_time: float
    resolution: str = Resolution.I_SLOW.value
    my_new_speed: float = 0.0
    my_priority: list[float] = field(default_factory=list)


@dataclass
class WaitFor:
    """One edge of the distributed deadlock dependency graph."""
    header: Header
    waiting_for: int = 0              # robot_id blocking me, 0 = none
    blocked_since: float = 0.0
    blocking_cx: int = -1
    blocking_cy: int = -1
    # -- ADDED in schema 1.1.0, additive only: every field below defaults, so
    # a 1.0.0 receiver decodes a 1.1.0 WaitFor unchanged. Flagged here rather
    # than edited silently, per the frozen-contract rule.
    #
    # These three are the input to LIFO deadlock resolution. A cycle is
    # broken by the robot that entered the contested region LAST, and that
    # cannot be computed from position alone -- it needs the entry stamp,
    # which only the entering robot knows.
    entered_at: float = 0.0           # when I entered my contested region
    segment_id: int = -1              # aisle segment I am contesting, -1 = none
    dist_to_block: float = 0.0        # metres from me to the robot blocking me
    backing_out: bool = False         # I am reversing along my own trail


# --------------------------------------------------------------------------
# Telemetry  (robot -> server -> dashboard).  MONITORING ONLY.
# --------------------------------------------------------------------------

@dataclass
class Telemetry:
    """
    Dashboard team: build against this and nothing else.

    Note there is no command channel back. The server observes; it does
    not control. That is the whole architectural claim.
    """
    header: Header
    state: RobotState
    intent: Intent
    peers_heard: list[int] = field(default_factory=list)
    comms_degraded: bool = False
    deadlock_flag: bool = False
    stops_count: int = 0
    distance_travelled: float = 0.0
    tasks_completed: int = 0
    cpu_percent: float = 0.0


# --------------------------------------------------------------------------
# LinkPacket  --  EVERY peer message travels inside one of these.
# --------------------------------------------------------------------------

@dataclass
class LinkPacket:
    """
    Uniform envelope for all peer-to-peer traffic.

    Why one envelope instead of a topic per message type: the comms
    mediator drops ALL peer traffic uniformly. With separate channels you
    will forget one, and that channel silently leaks information between
    robots -- which quietly invalidates the decentralisation claim.
    """
    src: int
    dst: int                          # BROADCAST(0) or specific robot_id
    msg_type: str
    payload: dict[str, Any]
    sent_at: float
    seq: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_json(s: str) -> "LinkPacket":
        return LinkPacket(**json.loads(s))

    def size_bytes(self) -> int:
        return len(self.to_json().encode())


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_seq_counters: dict[int, int] = {}


def next_seq(robot_id: int) -> int:
    _seq_counters[robot_id] = _seq_counters.get(robot_id, 0) + 1
    return _seq_counters[robot_id]


def make_header(robot_id: int, stamp: float | None = None,
                ttl_ms: int = 2000) -> Header:
    return Header(robot_id=robot_id,
                  seq=next_seq(robot_id),
                  stamp=time.time() if stamp is None else stamp,
                  ttl_ms=ttl_ms)


def wrap(msg: Any, msg_type: MsgType, src: int,
         dst: int = BROADCAST, now: float | None = None) -> LinkPacket:
    """Put any dataclass message into a LinkPacket envelope."""
    return LinkPacket(
        src=src,
        dst=dst,
        msg_type=msg_type.value,
        payload=asdict(msg),
        sent_at=time.time() if now is None else now,
        seq=msg.header.seq,
    )


def priority_tuple(priority_class: int, slack: float,
                   battery_soc: float, robot_id: int) -> list[float]:
    """
    Deterministic global priority (architecture doc section E.5).

    Compared LEXICOGRAPHICALLY, LOWER tuple = HIGHER priority.

    Every robot computes this for every peer from broadcast fields alone,
    so all robots agree on the ordering with zero negotiation round-trips.
    That agreement is what makes decentralised prioritised planning work.
    """
    return [float(priority_class),
            -float(slack),
            -float(1.0 - battery_soc),
            float(robot_id)]


def is_higher_priority(a: list[float], b: list[float]) -> bool:
    return tuple(a) < tuple(b)


__all__ = [
    "SCHEMA_VERSION", "BROADCAST", "INF_COST",
    "RobotMode", "MsgType", "ReleaseReason", "RerouteReason",
    "Resolution", "ObstacleKind",
    "Header", "RobotState", "Reservation", "Intent", "Task", "Bid",
    "Claim", "Release", "Obstacle", "Reroute", "Coordination", "WaitFor",
    "Telemetry", "LinkPacket",
    "next_seq", "make_header", "wrap", "priority_tuple", "is_higher_priority",
]
