"""
Fast headless 2D warehouse simulator.

WHY THIS EXISTS
---------------
Benchmarking needs 5 baselines x 20 seeds x minutes of scenario. In Gazebo
that is roughly 17 hours of wall clock and you will not get your number.
Here it runs in seconds, headless, deterministic per seed.

Gazebo is for the VISUAL demo with 3 robots. The coordination code is shared
between both -- only the sensor/actuator interface differs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from amr_msgs import (BROADCAST, INF_COST, Bid, Claim, Header, Intent, MsgType,
                      Release, ReleaseReason, RobotMode, RobotState, Task,
                      make_header, wrap)
from comms import CommsMediator
from coordination import (T_BID, T_CLAIM, T_COOLDOWN, T_REBID, T_RELEASE,
                          V_MIN, DeadlockDetector, adapt_speed, bid_cost,
                          conflict_risk, feasible, first_conflict, has_quorum,
                          make_priority, resolve_auction, sample_trajectory)
from planner import (ReservationTable, congestion_cost, path_length_m, plan,
                     path_to_reservations)
from warehouse_map import CELL_SIZE, WarehouseMap

DT = 0.1
V_MAX = 0.8
E_FULL = 100.0


@dataclass
class Metrics:
    collisions: int = 0
    near_misses: int = 0
    deadlocks: int = 0
    deadlock_recovery_s: list[float] = field(default_factory=list)
    tasks_completed: int = 0
    task_times: list[float] = field(default_factory=list)
    full_stops: int = 0
    time_stopped: float = 0.0
    distance: float = 0.0
    replans: int = 0
    sim_time: float = 0.0

    def summary(self) -> dict:
        n = max(1, len(self.task_times))
        return {
            "tasks_completed": self.tasks_completed,
            "avg_task_time": round(sum(self.task_times) / n, 2),
            "total_time": round(self.sim_time, 2),
            "throughput_per_min": round(60.0 * self.tasks_completed
                                        / max(1e-6, self.sim_time), 2),
            "collisions": self.collisions,
            "near_misses": self.near_misses,
            "deadlocks": self.deadlocks,
            "full_stops": self.full_stops,
            "time_stopped": round(self.time_stopped, 2),
            "distance_m": round(self.distance, 1),
            "replans": self.replans,
        }


class Robot:
    """One AMR. Identical software on every instance."""

    def __init__(self, rid: int, cx: int, cy: int, wmap: WarehouseMap,
                 comms: CommsMediator, mode_flags: dict):
        self.id = rid
        self.wmap = wmap
        self.comms = comms
        self.flags = mode_flags

        self.x, self.y = wmap.to_world(cx, cy)
        self.theta = 0.0
        self.v = 0.0
        self.v_nom = 0.6
        self.battery = 1.0
        self.capacity = 20.0
        self.payload = 0.0
        self.mode = RobotMode.IDLE

        self.task: Task | None = None
        self.goal: tuple[int, int] | None = None
        self.path: list[tuple[int, int]] = []
        self.path_idx = 0
        self.task_start_t = 0.0
        self.phase = "none"           # "to_pickup" | "to_dropoff"

        self.table = ReservationTable()
        self.peers: dict[int, RobotState] = {}
        self.peer_intents: dict[int, Intent] = {}
        self.peer_seen_at: dict[int, float] = {}
        self.deadlock = DeadlockDetector()

        self.yielding = False
        self.saved_goal: tuple[int, int] | None = None
        self.yield_deadline = 0.0
        self.degraded = False
        self.stops = 0
        self.stopped_time = 0.0
        self.dist = 0.0
        self.replans = 0
        self.blocked_since = 0.0
        self.last_broadcast = -1.0
        # -- auction state. All of it is built from RECEIVED packets only.
        self.pending_bids: dict[int, dict[int, tuple[float, bool]]] = {}
        self.known_tasks: dict[int, Task] = {}
        self.bid_close: dict[int, float] = {}    # task -> window shuts at
        self.claimed: dict[int, int] = {}        # task -> believed owner
        self.claim_due: dict[int, float] = {}    # task -> Claim deadline
        self.rebid_at: dict[int, float] = {}     # task -> next retry
        self.excluded: dict[int, set[int]] = {}  # task -> silent winners
        self.cooldown: dict[int, float] = {}     # task -> don't re-bid until
        self.stall_since: float | None = None    # self-measured, not told

    # -- geometry ----------------------------------------------------------

    @property
    def cell(self) -> tuple[int, int]:
        return self.wmap.to_cell(self.x, self.y)

    def priority(self, now: float) -> list[float]:
        pc = self.task.priority_class if self.task else 5
        dl = self.task.deadline if self.task else 0.0
        eta = now + self.remaining_distance() / max(0.05, self.v_nom)
        return make_priority(pc, dl, eta, self.battery, self.id)

    def remaining_distance(self) -> float:
        return max(0, len(self.path) - self.path_idx) * CELL_SIZE

    # -- communication -----------------------------------------------------

    def broadcast(self, now: float) -> None:
        if self.degraded:
            return
        st = RobotState(header=make_header(self.id, now), x=self.x, y=self.y,
                        theta=self.theta, vx=self.v * math.cos(self.theta),
                        vy=self.v * math.sin(self.theta),
                        mode=self.mode.value, battery_soc=self.battery,
                        payload_kg=self.payload, capacity_kg=self.capacity,
                        current_task=self.task.task_id if self.task else 0)
        self.comms.send(wrap(st, MsgType.ROBOT_STATE, self.id, BROADCAST, now), now)

        res = path_to_reservations(self.path[self.path_idx:], now, self.v_nom)
        it = Intent(header=make_header(self.id, now),
                    task_id=self.task.task_id if self.task else 0,
                    path_cells=[list(c) for c in self.path[self.path_idx:]],
                    reservations=res, nominal_speed=self.v_nom,
                    priority=self.priority(now))
        self.comms.send(wrap(it, MsgType.INTENT, self.id, BROADCAST, now), now)

    def receive(self, now: float) -> list:
        tasks = []
        for pkt in self.comms.receive(self.id):
            t = pkt.msg_type
            if t == MsgType.ROBOT_STATE.value:
                d = dict(pkt.payload)
                d["header"] = Header(**d["header"])
                self.peers[pkt.src] = RobotState(**d)
                self.peer_seen_at[pkt.src] = now
            elif t == MsgType.INTENT.value:
                d = dict(pkt.payload)
                d["header"] = Header(**d["header"])
                from amr_msgs import Reservation
                d["reservations"] = [Reservation(**r) for r in d["reservations"]]
                it = Intent(**d)
                self.peer_intents[pkt.src] = it
                self.table.update(pkt.src, it.reservations, now)
            elif t == MsgType.TASK.value:
                d = dict(pkt.payload)
                d["header"] = Header(**d["header"])
                task = Task(**d)
                tasks.append(task)
                if task.task_id not in self.known_tasks:
                    self.known_tasks[task.task_id] = task
                    self.submit_bid(task, now)
            elif t == MsgType.BID.value:
                p = pkt.payload
                self.pending_bids.setdefault(p["task_id"], {})[pkt.src] = (
                    p["cost"], p["feasible"])
            elif t == MsgType.CLAIM.value:
                self._on_claim(pkt.src, pkt.payload["task_id"], now)
            elif t == MsgType.RELEASE.value:
                self._on_release(pkt.src, pkt.payload["task_id"], now)
        return tasks

    def check_degraded(self, now: float) -> None:
        """Comms health -> degraded mode (doc section G)."""
        fresh = [r for r, t in self.peer_seen_at.items() if now - t < 2.0]
        was = self.degraded
        self.degraded = (len(fresh) == 0 and len(self.comms.ids) > 1
                         and now > 3.0)
        if self.degraded:
            self.v_nom = min(self.v_nom, 0.4 * V_MAX)   # slow down
            for rid in list(self.table._res):            # drop stale peers
                if now - self.peer_seen_at.get(rid, 0) > 4.0:
                    self.table.drop(rid)
        elif was:
            self.v_nom = 0.6

    # -- task allocation: consensus sealed-bid auction (doc 0.3, E.1) ------
    #
    # There is no auctioneer. Each robot evaluates locally, BROADCASTS its
    # bid, listens for T_BID, and runs resolve_auction() over the bids it
    # actually received. On a lossy link two robots can therefore see
    # different bid sets and both believe they won; the Claim exchange is
    # what collapses that back to a single owner.

    def submit_bid(self, task: Task, now: float) -> None:
        """Evaluate locally and put the bid on the wire."""
        cost, ok = self.evaluate_task(task, now)
        self.pending_bids.setdefault(task.task_id, {})[self.id] = (cost, ok)
        self.bid_close[task.task_id] = now + T_BID
        self.rebid_at.pop(task.task_id, None)
        bid = Bid(header=make_header(self.id, now), task_id=task.task_id,
                  cost=cost, feasible=ok)
        self.comms.send(wrap(bid, MsgType.BID, self.id, BROADCAST, now), now)
        if ok and self.task is None:
            self.mode = RobotMode.BIDDING

    def step_auction(self, now: float) -> None:
        """Close due bid windows, claim wins, and chase silent winners."""
        for tid, close in list(self.bid_close.items()):
            if now < close:
                continue
            self.bid_close.pop(tid, None)
            # exclusion lasts exactly one resolution round (doc 0.3 step 5)
            ruled_out = self.excluded.pop(tid, set())
            if tid in self.claimed:
                continue
            bids = {r: v for r, v in self.pending_bids.get(tid, {}).items()
                    if r not in ruled_out}
            winner = resolve_auction(bids)
            task = self.known_tasks.get(tid)

            # NO QUORUM, NO COMMIT. Hearing only yourself is not winning an
            # auction, it is being alone. Without this a blackout makes every
            # robot the sole bidder and all of them take the same task.
            if winner == self.id and not has_quorum(len(bids),
                                                    len(self.comms.ids)):
                self.claim_due.pop(tid, None)
                self.rebid_at[tid] = now + T_REBID
                if self.task is None and self.mode == RobotMode.BIDDING:
                    self.mode = RobotMode.IDLE
                continue

            if winner == self.id and self.task is None and task is not None:
                self.claimed[tid] = self.id
                clm = Claim(header=make_header(self.id, now), task_id=tid,
                            winning_cost=bids[self.id][0])
                self.comms.send(
                    wrap(clm, MsgType.CLAIM, self.id, BROADCAST, now), now)
                self.accept_task(task, now)
                continue
            if winner and winner != self.id:
                self.claim_due[tid] = now + T_CLAIM
            else:
                self.rebid_at[tid] = now + T_REBID
            if self.task is None and self.mode == RobotMode.BIDDING:
                self.mode = RobotMode.IDLE

        # believed winner never announced -> rule it out and re-run
        for tid, due in list(self.claim_due.items()):
            if now < due:
                continue
            self.claim_due.pop(tid, None)
            if tid in self.claimed:
                continue
            presumed = resolve_auction(self.pending_bids.get(tid, {}))
            if presumed:
                self.excluded.setdefault(tid, set()).add(presumed)
            self.rebid_at[tid] = now

        # retry tasks nobody could take yet (every robot busy or infeasible)
        for tid, when in list(self.rebid_at.items()):
            if now < when:
                continue
            if tid in self.claimed or tid not in self.known_tasks:
                self.rebid_at.pop(tid, None)
                continue
            if now < self.cooldown.get(tid, 0.0):
                continue          # I just gave this one back -- let a peer try
            self.pending_bids.pop(tid, None)
            self.submit_bid(self.known_tasks[tid], now)

    def check_release(self, now: float) -> None:
        """
        Give a task back if I have been stalled too long trying to FETCH it.

        The fleet previously had no way to undo an allocation: a robot wedged
        in a deadlock held its task forever while peers that could have done
        it sat idle and the task never returned to auction. `ReleaseReason`
        declared BLOCKED for exactly this and nothing ever used it.

        Only ever released in the `to_pickup` phase. Once loaded the robot is
        physically holding the pallet, so handing the task to a peer would be
        a lie -- a carrying robot must finish or be unloaded by a human.
        """
        if self.task is None or self.yielding or self.phase != "to_pickup":
            return
        if self.stall_since is None or now - self.stall_since < T_RELEASE:
            return
        tid = self.task.task_id
        self.cooldown[tid] = now + T_COOLDOWN
        self.claimed.pop(tid, None)
        self.release_task(tid, now, ReleaseReason.BLOCKED)
        self.stall_since = None

    def _on_claim(self, src: int, tid: int, now: float) -> None:
        """
        Record a peer's claim, and break a double-claim deterministically.

        Packet loss makes two winners possible, because each robot resolved
        over a different bid set. Lowest robot_id keeps the task and the
        other releases -- both sides compute the same answer from the same
        rule, so there is no negotiation round-trip.
        """
        prev = self.claimed.get(tid)
        self.claimed[tid] = src if prev is None else min(prev, src)
        self.bid_close.pop(tid, None)
        self.claim_due.pop(tid, None)
        self.rebid_at.pop(tid, None)
        if (self.task is not None and self.task.task_id == tid
                and self.claimed[tid] != self.id):
            self.release_task(tid, now, ReleaseReason.PREEMPTED)

    def _on_release(self, src: int, tid: int, now: float) -> None:
        """A peer gave a task back -- reopen bidding on it."""
        if self.claimed.get(tid) == src:
            self.claimed.pop(tid, None)
            if tid in self.known_tasks:
                self.rebid_at[tid] = now

    def release_task(self, tid: int, now: float,
                     reason: ReleaseReason) -> None:
        """Drop a task and tell the fleet, so it can be re-auctioned."""
        rel = Release(header=make_header(self.id, now), task_id=tid,
                      reason=reason.value)
        self.comms.send(wrap(rel, MsgType.RELEASE, self.id, BROADCAST, now),
                        now)
        self.task = None
        self.goal = None
        self.path = []
        self.path_idx = 0
        self.payload = 0.0
        self.phase = "none"
        self.mode = RobotMode.IDLE

    def evaluate_task(self, task: Task, now: float) -> tuple[float, bool]:
        if self.task is not None or self.battery < 0.25:
            return INF_COST, False
        cx, cy = self.cell
        p1 = plan(self.wmap, (cx, cy), (task.pickup_cx, task.pickup_cy),
                  self.table, self.id, now, now, self.v_nom, V_MAX,
                  self.flags["congestion"])
        if not p1:
            return INF_COST, False
        p2 = plan(self.wmap, (task.pickup_cx, task.pickup_cy),
                  (task.dropoff_cx, task.dropoff_cy), self.table, self.id,
                  now, now, self.v_nom, V_MAX, self.flags["congestion"])
        if not p2:
            return INF_COST, False

        dist = path_length_m(p1) + path_length_m(p2)
        eta = dist / self.v_nom
        energy = dist * 0.35
        if not feasible(self.capacity, self.payload, task.payload_kg,
                        self.battery, energy, E_FULL):
            return INF_COST, False

        cong = (congestion_cost(p1, self.table, self.id, now, now, self.v_nom)
                if self.flags["congestion"] else 0.0)
        risk = 0.0
        if self.flags["congestion"]:
            mine = sample_trajectory(p1, now, self.v_nom)
            others = {r: sample_trajectory(
                [tuple(c) for c in i.path_cells], now, max(0.05, i.nominal_speed))
                for r, i in self.peer_intents.items()}
            risk = conflict_risk(mine, others)

        return bid_cost(eta, cong, energy, self.battery * E_FULL, 0, risk), True

    def accept_task(self, task: Task, now: float) -> None:
        self.task = task
        self.task_start_t = now
        self.phase = "to_pickup"
        self.mode = RobotMode.TO_PICKUP
        self.goal = (task.pickup_cx, task.pickup_cy)
        self.replan(now)

    def replan(self, now: float) -> bool:
        if self.goal is None:
            return False
        cx, cy = self.cell
        p = plan(self.wmap, (cx, cy), self.goal, self.table, self.id,
                 now, now, self.v_nom, V_MAX, self.flags["congestion"])
        if p:
            self.path, self.path_idx = p, 0
            self.replans += 1
            return True
        return False

    # -- motion ------------------------------------------------------------

    def corridor_blocked(self, now: float) -> bool:
        """
        Corridor reservation (architecture doc 0.4).

        Refuse to ENTER a narrow aisle segment that a peer is already
        traversing in the opposite direction. Holding at the mouth costs a
        few seconds; meeting head-on inside costs a deadlock, because no
        velocity solution exists in a one-robot-wide corridor.

        Crucially this only ever blocks ENTRY -- a robot already inside is
        never stopped by this rule, so the rule itself cannot deadlock.
        """
        if self.path_idx >= len(self.path):
            return False
        here = self.wmap.aisle_at(*self.cell)
        nxt_cell = self.path[self.path_idx]
        nxt = self.wmap.aisle_at(*nxt_cell)
        if nxt == -1 or nxt == here:
            return False                  # not entering a new corridor

        my_dir = nxt_cell[1] - self.cell[1]
        occupants = 0
        for rid, st in self.peers.items():
            if now - self.peer_seen_at.get(rid, 0) > 2.0:
                continue
            pc = self.wmap.to_cell(st.x, st.y)
            if self.wmap.aisle_at(*pc) != nxt:
                continue
            occupants += 1

            # Infer peer heading. Its broadcast path starts at its CURRENT
            # cell, so look ahead to the first cell that actually differs.
            their_dir = 0
            if abs(st.vy) > 0.05:
                their_dir = 1 if st.vy > 0 else -1
            else:
                it = self.peer_intents.get(rid)
                if it:
                    for c in it.path_cells[:6]:
                        if c[1] != pc[1]:
                            their_dir = 1 if c[1] > pc[1] else -1
                            break

            if my_dir != 0 and their_dir != 0 and my_dir * their_dir < 0:
                return True                        # head-on: do not enter

        # Capacity limit: a 3-cell-wide corridor cannot absorb a queue.
        # Holding at the mouth is cheap; gridlock inside is not.
        return occupants >= 2

    def decide_speed(self, now: float) -> float:
        """
        Speed adaptation (doc E.6). This is the mechanism that beats
        stop-and-wait: compute an arrival time that clears the conflict and
        derive the speed for it, rather than halting.
        """
        # corridor reservation applies to BOTH arms (it is a safety rule,
        # not the mechanism under test)
        if self.corridor_blocked(now):
            return 0.0

        if not self.flags["speed_adapt"]:
            return self._stop_and_wait_speed(now)

        mine = sample_trajectory(self.path[self.path_idx:], now, self.v_nom)
        if not mine:
            return self.v_nom
        others = {}
        for rid, it in self.peer_intents.items():
            if now - self.peer_seen_at.get(rid, 0) > 2.0:
                continue
            others[rid] = sample_trajectory(
                [tuple(c) for c in it.path_cells], now,
                max(0.05, it.nominal_speed))

        hit = first_conflict(mine, others)
        if hit is None:
            return V_MAX if self.flags["speed_adapt"] else self.v_nom

        t_c, pid, _, _ = hit
        my_p = self.priority(now)
        peer_p = self.peer_intents[pid].priority or [1e9]
        if tuple(my_p) < tuple(peer_p):
            return V_MAX                    # I have right of way

        dist = max(0.1, (t_c - now) * self.v_nom)
        peer_exit = t_c + 1.0
        v_new, action = adapt_speed(dist, peer_exit, now, self.v,
                                    V_MIN, V_MAX)
        if action == "STOP" and self.v > 0.01:
            self.stops += 1          # count TRANSITIONS into stop, not ticks
        return v_new

    def _stop_and_wait_speed(self, now: float) -> float:
        """BASELINE B0. Full stop on any predicted conflict."""
        mine = sample_trajectory(self.path[self.path_idx:], now, self.v_nom)
        others = {}
        for rid, it in self.peer_intents.items():
            if now - self.peer_seen_at.get(rid, 0) > 2.0:
                continue
            others[rid] = sample_trajectory(
                [tuple(c) for c in it.path_cells], now,
                max(0.05, it.nominal_speed))
        hit = first_conflict(mine, others)
        if hit is None:
            return self.v_nom
        _, pid, _, _ = hit
        my_p = self.priority(now)
        peer_p = self.peer_intents[pid].priority or [1e9]
        if tuple(my_p) < tuple(peer_p):
            return self.v_nom
        if self.v > 0.01:
            self.stops += 1              # transition into stop
        return 0.0                       # halt for the whole conflict

    def orca_adjust(self, hx: float, hy: float,
                    sensed: list[tuple[float, float]]) -> tuple[float, float]:
        """
        L3. Simplified reciprocal avoidance (ORCA-style lateral sidestep).

        ZONE-GATED, per architecture doc 0.4: active in OPEN areas only.
        In a narrow aisle the feasible velocity set for a head-on encounter
        is empty -- no reciprocal velocity method can solve it, so we bypass
        ORCA there and let the yield/reroute layer handle it instead.

        Reciprocity comes from a shared convention: everyone sidesteps to
        their own RIGHT. Both robots deviate in opposite world directions,
        so they slide past instead of mutually braking to a livelock.
        """
        cx, cy = self.cell
        if self.wmap.is_narrow(cx, cy):
            return hx, hy                      # bypass: not ORCA's domain

        R_AVOID = 1.8
        push_x = push_y = 0.0
        for (px, py) in sensed:
            dx, dy = px - self.x, py - self.y
            d = math.hypot(dx, dy)
            if d < 1e-6 or d > R_AVOID:
                continue
            # only react to peers roughly ahead of us
            if (dx * hx + dy * hy) / d < 0.2:
                continue
            # unit vector to my right of current heading
            rx, ry = hy, -hx
            strength = (R_AVOID - d) / R_AVOID
            push_x += rx * strength * 1.6
            push_y += ry * strength * 1.6

        nx, ny = hx + push_x, hy + push_y
        n = math.hypot(nx, ny)
        if n < 1e-6:
            return hx, hy
        return nx / n, ny / n

    def safety_supervisor(self, v_cmd: float,
                          sensed: list[tuple[float, float]]) -> float:
        """
        L2. Deterministic geometric brake. FINAL AUTHORITY over speed.

        Two properties that matter:
          * It can only ever REDUCE v_cmd, never raise it. No learned
            component can talk its way past this.
          * It runs on LiDAR returns, NOT on comms. So it keeps working
            when the network is dead -- which is exactly why the fleet
            stays collision-free in degraded mode.
        """
        R_HARD, R_SLOW = 0.50, 1.05
        v_out = v_cmd
        for (px, py) in sensed:
            d = math.hypot(px - self.x, py - self.y)
            if d < R_HARD:
                return 0.0                       # hard stop
            if d < R_SLOW:
                # scale linearly between hard-stop and slow radius
                scale = (d - R_HARD) / (R_SLOW - R_HARD)
                v_out = min(v_out, v_cmd * scale)
        return max(0.0, v_out)

    def step(self, now: float, dt: float,
             sensed: list[tuple[float, float]] | None = None) -> None:
        if not self.path or self.path_idx >= len(self.path):
            self.v = 0.0
            return

        v_cmd = self.decide_speed(now)
        # L2 has the last word, always.
        self.v = self.safety_supervisor(v_cmd, sensed or [])
        # Measure my own stall. Do not wait to be told by a central observer.
        if self.v < 0.05:
            if self.stall_since is None:
                self.stall_since = now
        else:
            self.stall_since = None
        if self.v < 0.01:
            self.stopped_time += dt
            return

        tx, ty = self.wmap.to_world(*self.path[self.path_idx])
        dx, dy = tx - self.x, ty - self.y
        d = math.hypot(dx, dy)
        if d < 0.12:
            self.path_idx += 1
            return

        hx, hy = dx / d, dy / d
        hx, hy = self.orca_adjust(hx, hy, sensed or [])   # L3

        move = min(d, self.v * dt)
        nx, ny = self.x + move * hx, self.y + move * hy
        # never drive into structure
        if not self.wmap.is_free(*self.wmap.to_cell(nx, ny)):
            nx, ny = self.x + move * (dx / d), self.y + move * (dy / d)
            if not self.wmap.is_free(*self.wmap.to_cell(nx, ny)):
                self.stopped_time += dt
                return

        self.theta = math.atan2(ny - self.y, nx - self.x)
        self.dist += math.hypot(nx - self.x, ny - self.y)
        self.x, self.y = nx, ny
        self.battery = max(0.0, self.battery - move * 0.0008)


class Simulation:
    def __init__(self, n_robots: int = 4, seed: int = 0,
                 congestion: bool = True, speed_adapt: bool = True,
                 loss_rate: float = 0.0, use_policy: bool = False):
        self.rng = random.Random(seed)
        self.wmap = WarehouseMap()
        ids = list(range(1, n_robots + 1))
        self.comms = CommsMediator(ids, random.Random(seed),
                                   loss_rate=loss_rate)
        flags = {"congestion": congestion, "speed_adapt": speed_adapt,
                 "policy": use_policy}

        starts = [(2, 2), (37, 2), (2, 27), (37, 27), (20, 2)][:n_robots]
        self.robots = [Robot(i, *starts[i - 1], self.wmap, self.comms, flags)
                       for i in ids]

        self.t = 0.0
        self.metrics = Metrics()
        self.tasks: list[Task] = []
        self.next_task_id = 1
        self.open_tasks: dict[int, Task] = {}
        self.task_announced_at: dict[int, float] = {}

    def announce_task(self) -> None:
        nodes = self.wmap.nodes
        pick = self.rng.choice([n for n in nodes.values() if n.kind == "pickup"])
        drop = self.rng.choice([n for n in nodes.values() if n.kind == "dropoff"])
        t = Task(header=make_header(0, self.t), task_id=self.next_task_id,
                 pickup_cx=pick.cx, pickup_cy=pick.cy,
                 dropoff_cx=drop.cx, dropoff_cy=drop.cy,
                 payload_kg=self.rng.uniform(1, 10),
                 announced_at=self.t)
        self.next_task_id += 1
        self.open_tasks[t.task_id] = t
        self.task_announced_at[t.task_id] = self.t
        for r in self.robots:
            self.comms._inbox[r.id].append(
                wrap(t, MsgType.TASK, 0, r.id, self.t))

    def check_collisions(self) -> None:
        for i, a in enumerate(self.robots):
            for b in self.robots[i + 1:]:
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < 0.44:
                    self.metrics.collisions += 1
                elif d < 0.8:
                    self.metrics.near_misses += 1

    def step(self) -> None:
        self.t += DT
        self.comms.step(self.t)

        for r in self.robots:
            self.comms.update_position(r.id, r.x, r.y)
            r.receive(self.t)
            r.check_degraded(self.t)
            if self.t - r.last_broadcast > 0.2:
                r.broadcast(self.t)
                r.last_broadcast = self.t

        # Each robot resolves the auction from its OWN received bids. The
        # server never picks a winner -- it only observes, from telemetry,
        # which tasks have been taken, so it knows what is still open.
        for r in self.robots:
            r.check_release(self.t)
            r.step_auction(self.t)
        held = {r.task.task_id for r in self.robots if r.task}
        for tid in list(self.open_tasks):
            if tid in held:
                self.open_tasks.pop(tid)
        # a task handed back by a stalled robot is open again (telemetry only --
        # the robots re-auction it among themselves regardless)
        for r in self.robots:
            for tid in r.known_tasks:
                if tid not in held and tid not in self.open_tasks \
                        and tid < self.next_task_id and r.cooldown.get(tid):
                    self.open_tasks[tid] = r.known_tasks[tid]

        for r in self.robots:
            # Simulated LiDAR: geometric detection of anything within range.
            # Deliberately independent of comms -- this is local sensing.
            sensed = [(o.x, o.y) for o in self.robots
                      if o.id != r.id
                      and math.hypot(o.x - r.x, o.y - r.y) < 3.0]
            r.step(self.t, DT, sensed)

            # yield complete -> resume the original goal
            if r.yielding and (r.path_idx >= len(r.path)
                               or self.t > r.yield_deadline):
                r.goal = r.saved_goal
                r.saved_goal = None
                r.yielding = False
                r.blocked_since = 0.0
                if r.goal:
                    r.replan(self.t)
                self.metrics.deadlock_recovery_s.append(DT)
                continue

            if r.task and not r.yielding and r.path_idx >= len(r.path) and r.path:
                if r.phase == "to_pickup":
                    r.phase = "to_dropoff"
                    r.mode = RobotMode.TO_DROPOFF
                    r.payload = r.task.payload_kg
                    r.goal = (r.task.dropoff_cx, r.task.dropoff_cy)
                    r.replan(self.t)
                else:
                    self.metrics.tasks_completed += 1
                    # Measure from ANNOUNCEMENT, not from the last accept.
                    # task_start_t resets every time a task is accepted, so a
                    # task released by a stalled robot and picked up by a peer
                    # would only be timed from the final successful attempt --
                    # the failed attempt would vanish from the metric. Timing
                    # from announced_at is the operator's truth and cannot be
                    # gamed by re-allocation. It also folds in allocation
                    # latency, which was previously excluded.
                    self.metrics.task_times.append(self.t - r.task.announced_at)
                    r.task, r.path = None, []
                    r.payload, r.phase = 0.0, "none"
                    r.mode = RobotMode.IDLE
                    # Clear the aisle. An idle robot parked on a dropoff
                    # blocks every peer routed there.
                    park = min(
                        [n for n in self.wmap.nodes.values()
                         if n.kind == "charger"],
                        key=lambda n: abs(n.cx - r.cell[0]) + abs(n.cy - r.cell[1]))
                    r.goal = (park.cx, park.cy)
                    r.replan(self.t)

        self.resolve_deadlocks()
        self.check_collisions()

    def resolve_deadlocks(self) -> None:
        """
        Doc E.7. Detection via wait-for cycle, resolution by deterministic
        priority: the LOWEST-priority member of the cycle yields into the
        nearest passing bay. Every robot computes the same yielder from the
        same broadcast data, so no negotiation round-trip is needed.
        """
        stalled = {}
        for r in self.robots:
            if r.path and r.path_idx < len(r.path) and r.v < 0.05:
                r.blocked_since = r.blocked_since or self.t
                if self.t - r.blocked_since > 3.0:
                    stalled[r.id] = r
            else:
                r.blocked_since = 0.0

        if not stalled:
            return

        # build wait-for edges: I am blocked by whoever is nearest ahead
        det = DeadlockDetector()
        for rid, r in stalled.items():
            nearest, best_d = 0, 1e9
            for o in self.robots:
                if o.id == rid:
                    continue
                d = math.hypot(o.x - r.x, o.y - r.y)
                if d < 1.5 and d < best_d:
                    nearest, best_d = o.id, d
            det.set_edge(rid, nearest)

        cycle = det.find_cycle()
        prios = {r.id: r.priority(self.t) for r in self.robots}

        if cycle:
            yielder_id = DeadlockDetector.choose_yielder(cycle, prios)
            r = next(x for x in self.robots if x.id == yielder_id)
            if not r.yielding and r.goal:
                bay = self.wmap.nearest_bay(*r.cell)
                if bay and bay != r.cell:
                    self.metrics.deadlocks += 1
                    r.saved_goal = r.goal        # remember the real goal
                    r.goal = bay
                    r.yielding = True
                    r.yield_deadline = self.t + 15.0
                    r.replan(self.t)
                    r.blocked_since = 0.0
        else:
            # no cycle: single blockage. Mark it and route around.
            for rid, r in stalled.items():
                if self.t - r.blocked_since > 5.0:
                    if r.path_idx < len(r.path):
                        bx, by = r.path[r.path_idx]
                        r.table.mark_blocked(bx, by, self.t)
                    r.replan(self.t)
                    r.blocked_since = 0.0

    def run(self, duration: float = 180.0, task_interval: float = 8.0,
            max_tasks: int = 20) -> dict:
        next_task = 1.0
        while self.t < duration:
            if self.t >= next_task and self.next_task_id <= max_tasks:
                self.announce_task()
                next_task = self.t + task_interval
            self.step()
            if (self.metrics.tasks_completed >= max_tasks
                    and not self.open_tasks):
                break

        self.metrics.sim_time = self.t
        self.metrics.full_stops = sum(r.stops for r in self.robots)
        self.metrics.time_stopped = sum(r.stopped_time for r in self.robots)
        self.metrics.distance = sum(r.dist for r in self.robots)
        self.metrics.replans = sum(r.replans for r in self.robots)
        out = self.metrics.summary()
        out["comms"] = self.comms.stats()
        return out


if __name__ == "__main__":
    sim = Simulation(n_robots=4, seed=1)
    import json
    print(json.dumps(sim.run(), indent=2))
