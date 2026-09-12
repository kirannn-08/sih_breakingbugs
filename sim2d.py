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

from amr_msgs import (BROADCAST, INF_COST, BayClaim, Bid, Claim, Coordination,
                      Header, Intent, MsgType,
                      Obstacle, ObstacleKind, Release, ReleaseReason, Resolution,
                      RobotMode, RobotState, Task, WaitFor, make_header, wrap)
from comms import CommsMediator
from features import deadlock_features
from learned import DeadlockRiskModel
from perception import (LIDAR_BLOCK_CONF, LidarTracker)
from coordination import (CHARGE_RATE, OBSTACLE_CONF, SOC_LOW, SOC_RESUME,
                          T_BACKOUT, T_BID, T_CLAIM, T_COOLDOWN, T_OBSTACLE,
                          T_REBID, T_RELEASE, T_WAITFOR, V_MIN,
                          DeadlockDetector, adapt_speed, bid_cost,
                          conflict_risk, feasible, first_conflict, has_quorum,
                          make_priority, predicted_head_on, resolve_auction,
                          resolve_pushback, sample_trajectory, segment_windows,
                          T_YIELD_CONFIRM)
from planner import (ReservationTable, congestion_cost, path_length_m, plan,
                     path_to_reservations, plan_st, schedule_of)
from warehouse_map import CELL_SIZE, WarehouseMap

DT = 0.1
V_MAX = 0.8
E_FULL = 100.0

# -- PHYSICAL FOOTPRINT -----------------------------------------------------
# There was previously no footprint in this file at all: collisions were
# tested with a bare `d < 0.44`, which is less than half the real robot's
# width. Two AMRs at 0.5 m centre-to-centre were counted as "no collision"
# while physically interpenetrating, which is why the fleet reported zero
# collisions while visibly overlapping on the dashboard.
ROBOT_LEN = 1.00                                   # m, along heading
ROBOT_WIDTH = 0.98                                 # m, wheels included
ROBOT_RADIUS = math.hypot(ROBOT_LEN / 2, ROBOT_WIDTH / 2)   # 0.70 m

D_COLLIDE = ROBOT_WIDTH                            # 0.98 m, bodies touching
D_NEAR = 2 * ROBOT_RADIUS                          # 1.40 m, swept circles touch

# Safety supervisor, DIRECTIONAL.
#
# An isotropic braking radius cannot work for a robot this size. Setting a
# hard-stop ring at 1.10 m means two AMRs may never be closer than 1.10 m,
# which forbids them passing side by side at all -- they pin each other at
# 1.11 m and crawl at 0.007 m/s forever. Real braking is about what is in
# your LANE ahead, not what is beside or behind you.
#
# So the check runs in the robot's body frame: a peer only matters if its
# lateral offset is small enough that we would actually clip it, and it is
# ahead of us. An omnidirectional floor still catches genuine contact.
R_HARD = ROBOT_LEN + 0.10                          # 1.10 m ahead, in-lane
R_SLOW = ROBOT_LEN + 1.30                          # 2.30 m ahead, in-lane
LANE_HALF_W = ROBOT_WIDTH                          # 0.98 m lateral clearance

# LiDAR. Range is the same 3.0 m the simulator always used; what is new is
# that returns are now OCCLUDED by structure and are fed to a track layer
# that the ROUTING and DEADLOCK decisions read, not just the brake.
LIDAR_RANGE = 3.0
T_LIDAR_REPLAN = 2.0        # min gap between LiDAR-triggered replans

# Blind-corner speed limit. Sight distance at an aisle mouth is about one
# body length, so V_BLIND is set to what can be stopped inside it: the brake
# resolves in one DT, and 0.30 m/s x the ~1.3 s a peer needs to emerge and
# be tracked keeps the contact distance above D_COLLIDE.
V_BLIND = 0.30
D_BLIND_LOOKAHEAD = 1.5     # m of path ahead scanned for a blind corner
T_BLIND_REACT = 1.5         # s of peer travel to treat as 'could be there'
SEG_TIE = 0.5               # entry times within this are a dead heat
MIN_BACKOUT_CELLS = 1       # a one-cell retreat still breaks a mutual stop
# Reverse as a general un-sticking tool, not only as deadlock recovery.
# Longer than T_OBSTACLE so cycle detection gets first refusal: a genuine
# deadlock should be resolved by the LIFO stack, and this is the fallback for
# everything else -- above all the shared-goal pile-up, where 3 robots want
# one dropoff cell, no wait-for CYCLE ever forms, and the losers simply sit.
T_PUSHBACK = 2.0            # stalled this long -> negotiate with the blocker
T_PUSHBACK_BC = 1.0         # request cadence
T_STUCK_REVERSE = 5.0
T_REVERSE_COOLDOWN = 8.0    # don't re-reverse immediately and oscillate

# Reversing is the blind manoeuvre, and it was being done at full speed into
# a region the robot cannot see well. All three residual collisions were a
# robot reversing at 0.800 m/s into a stationary peer on a DIAGONAL bearing,
# which slipped outside the forward lane width.
#
# Real AMRs reverse slowly for exactly this reason, and they watch a wider
# arc behind than in front. The rear cone must stay a cone and not become
# omnidirectional, though: a robot backs out precisely BECAUSE a peer is
# ~1.05 m in front of it, so an all-round hard stop would veto every backout
# at the instant it started and re-freeze the fleet.
V_REVERSE = 0.25              # m/s cap while backing up in the OPEN
# A lane is a straight run with racking on both sides, so nothing can enter
# it from the side: the only hazard behind you is a robot in the same lane,
# and the rear cone already covers that. Real AMRs are symmetric and drive
# either way along a straight aisle rather than executing a turn they have no
# room for. Crawling out of an aisle at 0.25 m/s was costing 228 s of
# fetch-phase stall across 6 runs -- 35% of all time stalled while fetching.
# Swept in closed loop, 10 seeds x 180 s, against a 30.5 s/fetch baseline:
#   0.25  25.0 s/fetch (+18.0%)  114 tasks  1 collision
#   0.35  23.2 s/fetch (+24.0%)  115 tasks  1 collision   <- chosen
#   0.45  21.7 s/fetch (+28.8%)  122 tasks  2 collisions
#   0.60  23.9 s/fetch (+21.5%)  110 tasks  4 collisions
# 0.60 is worse on BOTH axes than 0.45 -- past a point the reverse outruns
# the rear cone and the robot spends the gain on recovering from contact.
# 0.35 beats the 20% target while holding collisions at the slow-reverse
# level; 0.45 was not taken because one extra collision is not worth 4.8
# points of fetch time in a project whose headline claim is zero.
V_REVERSE_LANE = 0.35         # m/s when reversing inside a narrow aisle
LANE_HALF_W_REVERSE = 1.30    # wider arc behind: rear sensing is coarser
# TRIED AND REVERTED: a 0.06 m omnidirectional standoff while reversing, to
# buy the brake a tick of reaction before contact. It did NOT remove the
# residual graze (still 1 collision over 10 seeds) and cost 23% of lifelong
# throughput (75 -> 58 tasks over 400 s x 5). A margin that does not buy
# safety is just a slower robot.


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
    # -- instrumentation for the sensing/deadlock layers. Every new rule is
    # counted so a silent no-op cannot masquerade as working code.
    lidar_occluded: int = 0
    lidar_blocks: int = 0
    lidar_decisions: int = 0
    lidar_replans: int = 0
    blind_slowdowns: int = 0
    backouts: int = 0
    backouts_done: int = 0
    entry_deferrals: int = 0
    planned_waits: int = 0
    waits_executed: int = 0
    risk_deferrals: int = 0
    yield_skips: int = 0
    stuck_reverses: int = 0
    pushbacks: int = 0
    pushbacks_sent: int = 0
    bay_diversions: int = 0
    collision_ticks: int = 0
    near_miss_ticks: int = 0

    def summary(self) -> dict:
        n = max(1, len(self.task_times))
        return {
            "tasks_completed": self.tasks_completed,
            "avg_task_time": round(sum(self.task_times) / n, 2),
            "total_time": round(self.sim_time, 2),
            "throughput_per_min": round(60.0 * self.tasks_completed
                                        / max(1e-6, self.sim_time), 2),
            "collisions": self.collisions,
            "collision_ticks": self.collision_ticks,
            "near_misses": self.near_misses,
            "near_miss_ticks": self.near_miss_ticks,
            "deadlocks": self.deadlocks,
            "full_stops": self.full_stops,
            "time_stopped": round(self.time_stopped, 2),
            "distance_m": round(self.distance, 1),
            "replans": self.replans,
            "lidar_occluded": self.lidar_occluded,
            "lidar_blocks": self.lidar_blocks,
            "lidar_decisions": self.lidar_decisions,
            "lidar_replans": self.lidar_replans,
            "blind_slowdowns": self.blind_slowdowns,
            "backouts": self.backouts,
            "backouts_done": self.backouts_done,
            "entry_deferrals": self.entry_deferrals,
            "planned_waits": self.planned_waits,
            "waits_executed": self.waits_executed,
            "risk_deferrals": self.risk_deferrals,
            "yield_skips": self.yield_skips,
            "stuck_reverses": self.stuck_reverses,
            "pushbacks": self.pushbacks,
            "pushbacks_sent": self.pushbacks_sent,
            "bay_diversions": self.bay_diversions,
            "avg_recovery_s": round(
                sum(self.deadlock_recovery_s)
                / max(1, len(self.deadlock_recovery_s)), 2),
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
        self.charging = False                    # topping up on a bay
        self.home_bay: tuple[int, int] | None = None
        self.last_obstacle_bc = -1e9             # rate-limit announcements
        self.last_waitfor_bc = -1e9
        self.waitfor_target = 0                  # peer I believe blocks me
        # -- LOCAL SENSING. Independent of comms, by construction.
        self.lidar = LidarTracker()
        self.last_lidar_replan = -1e9
        self.lidar_blocks = 0        # cells marked blocked from sensing alone
        self.lidar_decisions = 0     # speed decisions driven by sensing alone
        self.lidar_replans = 0
        self.blind_slowdowns = 0
        self.peer_blocked_since: dict[int, float] = {}
        # -- CONTESTED-REGION STACK.
        # `trail` is the cells I have actually occupied, oldest first. It is
        # the only escape route guaranteed to be drivable and, for the last
        # robot into a corridor, guaranteed to be empty. `region_entered_at`
        # is my push timestamp; peers learn it from WaitFor and every robot
        # in a cycle then pops the same one.
        self.trail: list[tuple[int, int]] = [(cx, cy)]
        self.region = -1
        self.region_entered_at = 0.0
        self.backing_out = False
        self.backout_deadline = 0.0
        self.peer_entered_at: dict[int, float] = {}
        self.peer_dist_to_block: dict[int, float] = {}
        self.peer_backing_out: dict[int, bool] = {}
        # when I first computed peer X as the elected yielder. Used to time
        # out a candidate that never confirms it is backing out.
        self.yield_elect_since: dict[int, float] = {}
        self.yield_skips = 0
        self.last_stuck_reverse = -1e9
        self.stuck_reverses = 0
        # -- NEGOTIATED PUSH-BACK
        # Distance driven on the CURRENT leg. Progress is what decides who
        # reverses, and a peer cannot infer it -- RobotState carries position,
        # not odometry, and two robots a metre apart may have driven 0.5 m and
        # 30 m to get there.
        self.leg_start_dist = 0.0
        self.peer_progress: dict[int, float] = {}
        self.peer_prio: dict[int, list[float]] = {}
        self.pushback_seen: dict[int, float] = {}
        self.last_pushback_bc = -1e9
        self.pushbacks_sent = 0
        self.pushbacks_yielded = 0
        # -- BAY RESERVATION
        self.bay_claims: dict[tuple[int, int], tuple[int, float, float]] = {}
        self.my_bay_claim: tuple[int, int] | None = None
        self.last_bay_bc = -1e9
        self.bay_claim_stamp = 0.0
        self.bay_diversions = 0
        self.backouts = 0
        self.backouts_done = 0
        self.entry_deferrals = 0
        self.path_holds: list[float] = []
        self.planned_waits = 0        # holds the planner asked for
        self.waits_executed = 0       # ticks actually spent honouring one
        self.risk_deferrals = 0       # deferrals the MODEL caused, not the rule

    # -- geometry ----------------------------------------------------------

    @property
    def cell(self) -> tuple[int, int]:
        return self.wmap.to_cell(self.x, self.y)

    def priority(self, now: float) -> list[float]:
        pc = self.task.priority_class if self.task else 5
        dl = self.task.deadline if self.task else 0.0
        eta = now + self.remaining_distance() / max(0.05, self.v_nom)
        return make_priority(pc, dl, eta, self.battery, self.id)

    def progress_m(self) -> float:
        """Metres driven on the current leg. Reset when the leg changes."""
        return max(0.0, self.dist - self.leg_start_dist)

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
            elif t == MsgType.OBSTACLE.value:
                self._on_obstacle(pkt.src, pkt.payload, now)
            elif t == MsgType.COORDINATION.value:
                p = pkt.payload
                self.peer_prio[pkt.src] = list(p.get("my_priority") or [])
                self.peer_progress[pkt.src] = float(p.get("progress_m", 0.0))
                if int(p.get("conflict_with", 0)) == self.id:
                    self.pushback_seen[pkt.src] = now
            elif t == MsgType.BAY_CLAIM.value:
                p = pkt.payload
                bay = (int(p["bay_cx"]), int(p["bay_cy"]))
                if p.get("releasing"):
                    if self.bay_claims.get(bay, (0,))[0] == pkt.src:
                        self.bay_claims.pop(bay, None)
                else:
                    self.bay_claims[bay] = (pkt.src,
                                            float(p.get("claimed_at", now)),
                                            float(p.get("eta", 0.0)))
            elif t == MsgType.WAIT_FOR.value:
                p = pkt.payload
                self.deadlock.set_edge(pkt.src, int(p.get("waiting_for", 0)))
                self.peer_blocked_since[pkt.src] = float(p.get("blocked_since", 0.0))
                # Stack stamps (schema 1.1.0). A peer on an older schema omits
                # them and defaults to 0.0, which sorts as the OLDEST entrant --
                # so a robot that cannot report its stamp is never elected to
                # back out. Missing evidence must not volunteer a victim.
                self.peer_entered_at[pkt.src] = float(p.get("entered_at", 0.0))
                self.peer_dist_to_block[pkt.src] = float(
                    p.get("dist_to_block", 0.0)) or 1e9
                self.peer_backing_out[pkt.src] = bool(p.get("backing_out", False))
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

    # -- a stalled robot is an obstacle ------------------------------------

    def announce_obstacle(self, now: float) -> None:
        """
        Broadcast MYSELF as a blocking obstacle once I have been stalled long
        enough to be one.

        This is the piece that makes deadlock recoverable without a central
        referee. A wedged robot cannot free itself, but it CAN tell the fleet
        "this cell is blocked". Peers fold that into `blocked_belief`, which
        A* already weights by LAMBDA_BLOCK, so they route around instead of
        queueing behind it. The belief decays at 0.05/s, so it self-clears
        the moment the robot starts moving again -- no retraction message
        and no stale no-go zone.
        """
        if self.stall_since is None or now - self.stall_since < T_OBSTACLE:
            return
        if now - self.last_obstacle_bc < 1.0:
            return
        self.last_obstacle_bc = now
        cx, cy = self.cell
        cells = [[cx, cy]]
        if self.path_idx < len(self.path):          # also the cell I'm entering
            nx, ny = self.path[self.path_idx]
            if (nx, ny) != (cx, cy):
                cells.append([nx, ny])
        ob = Obstacle(header=make_header(self.id, now), cells=cells,
                      kind=ObstacleKind.DYNAMIC.value,
                      confidence=OBSTACLE_CONF, observed_at=now)
        self.comms.send(wrap(ob, MsgType.OBSTACLE, self.id, BROADCAST, now), now)

    def _on_obstacle(self, src: int, payload: dict, now: float) -> None:
        """A peer says it is blocking these cells. Believe it, and reroute."""
        cells = [tuple(c) for c in payload.get("cells", [])]
        conf = float(payload.get("confidence", 1.0))
        on_my_path = False
        for (cx, cy) in cells:
            self.table.mark_blocked(cx, cy, now, conf)
            if (cx, cy) in self.path[self.path_idx:]:
                on_my_path = True
        # Only replan if it actually affects me -- replanning the whole fleet
        # on every announcement is how you turn one stall into fleet-wide churn.
        if on_my_path and self.goal is not None:
            self.replan(now)

    def announce_waitfor(self, now: float) -> None:
        """
        Publish one edge of the fleet-wide wait-for graph.

        Every robot then builds the SAME graph from broadcasts and elects the
        same yielder locally, which is what `DeadlockDetector` was designed
        for. Previously the graph was assembled centrally by the simulation
        and this message was never sent at all.
        """
        if now - self.last_waitfor_bc < T_WAITFOR:
            return
        self.last_waitfor_bc = now
        # WHO IS BLOCKING ME = whoever the SAFETY SUPERVISOR is braking for.
        #
        # This used to test whether a peer was "ahead" by dotting (peer - me)
        # against (next waypoint - me). That is not the question the brake
        # asks, and at an intersection the two disagree: a peer stopping me
        # from a perpendicular bearing fails the dot product, so a robot that
        # was hard-stopped broadcast `waiting_for = 0` -- "nobody is blocking
        # me" -- while standing still because of that exact peer.
        #
        # Wait-for edges are the ONLY input to cycle detection, so a missing
        # edge is a missing cycle. Measured: NO_CYCLE on 1500 of 1500 ticks
        # with all four robots frozen, edges {3->2, 1->4}, a chain that can
        # never close. The whole LIFO stack was unreachable -- not broken,
        # never invoked.
        #
        # Now it uses the supervisor's own body-frame lane test, so if a peer
        # is the reason I am stopped, that peer is the one I name.
        blocker = 0
        if self.stall_since is not None and self.path_idx < len(self.path):
            hx, hy = math.cos(self.theta), math.sin(self.theta)
            best_d = 1e9
            for pid, st in self.peers.items():
                if now - self.peer_seen_at.get(pid, -1e9) > 2.0:
                    continue
                dx, dy = st.x - self.x, st.y - self.y
                d = math.hypot(dx, dy)
                fwd = dx * hx + dy * hy
                lat = -dx * hy + dy * hx
                in_lane = (fwd > 0.0 and abs(lat) < LANE_HALF_W
                           and fwd < R_SLOW)
                # Body contact counts whatever the bearing -- that is what the
                # omnidirectional floor brakes for.
                if (in_lane or d < D_NEAR) and d < best_d:
                    best_d, blocker = d, pid
        self.waitfor_target = blocker
        self.deadlock.set_edge(self.id, blocker)
        wf = WaitFor(header=make_header(self.id, now), waiting_for=blocker,
                     blocked_since=(self.stall_since or 0.0),
                     blocking_cx=self.cell[0], blocking_cy=self.cell[1],
                     entered_at=self.region_entered_at,
                     segment_id=self.region,
                     dist_to_block=(best_d if blocker else 0.0),
                     backing_out=self.backing_out)
        self.comms.send(wrap(wf, MsgType.WAIT_FOR, self.id, BROADCAST, now), now)

    def announce_pushback(self, now: float) -> None:
        """
        "You are in my way -- one of us must reverse." Sent to the peer the
        BRAKE is actually stopping me for (`waitfor_target`).

        Carries my priority and my progress so the receiver can evaluate the
        identical rule. Both sides reaching the same verdict from the same
        numbers is what stops the symmetric failure: previously each robot
        decided to reverse on its own evidence, so either both reversed or the
        one that did drove straight back into the same blockage.
        """
        if self.waitfor_target == 0 or self.stall_since is None:
            return
        if now - self.stall_since < T_PUSHBACK:
            return
        if now - self.last_pushback_bc < T_PUSHBACK_BC:
            return
        self.last_pushback_bc = now
        self.pushbacks_sent += 1
        mine = self.priority(now)
        theirs = self.peer_prio.get(self.waitfor_target)
        yielding = (resolve_pushback(mine, self.progress_m(), theirs,
                                     self.peer_progress.get(
                                         self.waitfor_target, 0.0),
                                     self.id, self.waitfor_target)
                    if theirs else False)
        msg = Coordination(
            header=make_header(self.id, now),
            conflict_with=self.waitfor_target,
            conflict_cx=self.cell[0], conflict_cy=self.cell[1],
            conflict_time=now,
            resolution=(Resolution.I_YIELD.value if yielding
                        else Resolution.I_PROCEED.value),
            my_new_speed=self.v, my_priority=mine,
            progress_m=self.progress_m(), request=True)
        self.comms.send(wrap(msg, MsgType.COORDINATION, self.id,
                             BROADCAST, now), now)

    def check_pushback(self, now: float) -> bool:
        """
        A peer asked me to move, or I asked it to. Decide, and reverse if I
        lost.

        Runs AFTER cycle detection so a genuine deadlock is still resolved by
        the LIFO stack, and BEFORE the unilateral stuck-reverse so a negotiated
        answer always beats a guessed one.

        The loser treats the winner's cell as an obstacle before retreating,
        so the replan after the reverse cannot simply re-derive the route it
        was just pushed off.
        """
        if self.backing_out or self.yielding or self.stall_since is None:
            return False
        if now - self.stall_since < T_PUSHBACK:
            return False
        if now - self.last_stuck_reverse < T_REVERSE_COOLDOWN:
            return False
        if self.at_bay() or self.charging:
            return False
        pid = self.waitfor_target
        # Either I am blocked by someone, or someone told me I am blocking.
        if pid == 0:
            recent = [(t, p) for p, t in self.pushback_seen.items()
                      if now - t < 2.0]
            if not recent:
                return False
            pid = max(recent)[1]
        theirs = self.peer_prio.get(pid)
        if theirs is None:
            return False        # no evidence yet; do not guess
        if not resolve_pushback(self.priority(now), self.progress_m(),
                                theirs, self.peer_progress.get(pid, 0.0),
                                self.id, pid):
            return False        # I win: hold my ground, they will move
        st = self.peers.get(pid)
        if st is not None:
            self.table.mark_blocked(*self.wmap.to_cell(st.x, st.y), now,
                                    OBSTACLE_CONF)
        if self.path_idx < len(self.path):
            self.table.mark_blocked(*self.path[self.path_idx], now,
                                    OBSTACLE_CONF)
        if not self.start_backout(now):
            return False
        self.last_stuck_reverse = now
        self.pushbacks_yielded += 1
        return True

    def check_deadlock(self, now: float) -> bool:
        """
        Detect a wait-for cycle from BROADCAST edges and decide locally
        whether I am the one who backs out.

        Resolution is LIFO: the cycle member that entered the contested
        region LAST pops off the stack and reverses along its own trail.
        `choose_yielder_lifo` is deterministic and every robot runs it on the
        same broadcast data, so all of them elect the same robot with no
        negotiation round-trip and no referee.

        Why this replaced "lowest priority yields": the old rule could elect
        a robot buried at the far end of an aisle whose only exit ran through
        the peer it was deadlocked with, so it retreated into a cell another
        cycle member needed and the jam survived. Measured on seed 3, nine
        detected deadlocks produced no recovery at all and a 2-cycle held for
        4589 robot-ticks. The last robot in is the one nearest the entrance,
        and the stretch behind it is the stretch it has just driven.
        """
        if not self.flags.get("lifo", True):
            return False            # B5 ablation: no stack, no backout
        if self.backing_out or self.yielding or self.stall_since is None:
            return False
        if now - self.stall_since < T_OBSTACLE:
            return False
        cycle = self.deadlock.find_cycle()
        if not cycle:
            self.yield_elect_since.clear()
            return False
        # The stack is the cycle plus everyone queued behind it: resolving a
        # jam means moving whoever is physically in the way, and that robot is
        # very often a mere waiter rather than a cycle member.
        stack = self.deadlock.blocked_set(cycle)
        if self.id not in stack:
            return False

        prios = {self.id: self.priority(now)}
        for pid, it in self.peer_intents.items():
            if it.priority:
                prios[pid] = list(it.priority)
        entered = dict(self.peer_entered_at)
        entered[self.id] = self.region_entered_at
        dists = dict(self.peer_dist_to_block)
        dists[self.id] = self.dist_to_blocker(now)

        # Walk the stack in pop order and take the first candidate that has
        # not already failed to move. Election is not ability: the top of the
        # stack may have a peer parked in its reverse path and no route out.
        # A single-candidate election has no fallback, and an elected robot
        # that cannot move freezes the whole fleet -- measured at 300 s of
        # total paralysis with the cycle correctly detected the whole time.
        #
        # A candidate is skipped once it has been elected for longer than
        # T_YIELD_CONFIRM without its broadcast `backing_out` flag going true.
        # That flag is already on the air, so every robot reaches the same
        # conclusion from the same evidence and the election stays
        # decentralised.
        order = DeadlockDetector.rank_yielders_lifo(stack, entered, dists,
                                                    prios)
        chosen = 0
        for cand in order:
            self.yield_elect_since.setdefault(cand, now)
            confirmed = (self.backing_out if cand == self.id
                         else self.peer_backing_out.get(cand, False))
            if confirmed:
                return False          # it is already moving; leave it alone
            if now - self.yield_elect_since[cand] > T_YIELD_CONFIRM:
                self.yield_skips += 1
                continue              # elected, never moved -> next one down
            chosen = cand
            break
        if chosen != self.id:
            return False              # someone else pops; hold position

        if self.start_backout(now):
            return True
        # I cannot move either. Mark myself as having had my turn so the
        # fleet stops waiting on me, rather than re-electing me every tick.
        self.yield_elect_since[self.id] = now - T_YIELD_CONFIRM - 1e-3
        return False

    def check_stuck_reverse(self, now: float) -> bool:
        """
        Stalled with no deadlock cycle to blame -> reverse and replan anyway.

        A robot only reversed as part of LIFO deadlock recovery, which needs a
        wait-for CYCLE. Most stalls here are not cycles: 60.6% of ticks have
        two or more robots routed to the same goal cell and 25% have three or
        more, because there are 3 dropoffs for 4 robots and a dropoff is one
        cell with no queue. The robots that lose that race are blocked by a
        chain, not a cycle, so the stack never fired and they sat still.

        Reversing is the right tool precisely because it needs no cooperation
        and no free space ahead: the robot retreats over ground it has already
        proved drivable, and the replan that follows starts from a different
        cell with the blockage now marked, so A* can pick a different approach
        instead of re-deriving the same blocked one.

        Never while parked on a charging bay -- a robot sitting on a bay is
        stationary because it has nothing to do, not because it is stuck, and
        reversing off the bay would put it back in an aisle for no reason.
        """
        if self.backing_out or self.yielding or self.stall_since is None:
            return False
        if now - self.stall_since < T_STUCK_REVERSE:
            return False
        if now - self.last_stuck_reverse < T_REVERSE_COOLDOWN:
            return False
        if self.at_bay() or self.charging or self.goal is None:
            return False
        if not self.path or self.path_idx >= len(self.path):
            return False
        # Mark what is in front of me before retreating, so the replan after
        # the reverse does not simply re-derive the route I am stuck on.
        bx, by = self.path[self.path_idx]
        self.table.mark_blocked(bx, by, now, OBSTACLE_CONF)
        if not self.start_backout(now):
            return False
        self.last_stuck_reverse = now
        self.stuck_reverses += 1
        return True

    def dist_to_waypoint(self) -> float:
        if self.path_idx >= len(self.path):
            return 0.0
        wx, wy = self.wmap.to_world(*self.path[self.path_idx])
        return max(0.05, math.hypot(wx - self.x, wy - self.y))

    def dist_to_blocker(self, now: float) -> float:
        """Metres to the peer I believe blocks me. Broadcast, and a LIFO tiebreak."""
        st = self.peers.get(self.waitfor_target)
        if st is None:
            return 1e9
        return math.hypot(st.x - self.x, st.y - self.y)

    def backout_target(self, now: float) -> list[tuple[int, int]] | None:
        """
        Reverse along my own trail to the first cell where a peer can pass me.

        Deliberately NOT a replan. A* would route me to that cell through
        whatever looks cheapest, which in a jam is straight back through the
        robots I am deadlocked with. My trail is the one route I know is
        physically drivable, and -- because I am the last robot into this
        region -- the one route no cycle member is standing in.
        """
        if len(self.trail) < 2:
            return None
        route: list[tuple[int, int]] = []
        for cell in reversed(self.trail[:-1]):
            wx, wy = self.wmap.to_world(*cell)
            occupied = any(
                math.hypot(st.x - wx, st.y - wy) < ROBOT_WIDTH
                for pid, st in self.peers.items()
                if now - self.peer_seen_at.get(pid, -1e9) < 2.0)
            if occupied:
                break                      # cannot reverse past a peer
            route.append(cell)
            if not self.wmap.is_narrow(*cell):
                return route               # wide enough to be passed here
        # Trail exhausted with no passing point. Retreat anyway if there is
        # anywhere at all to go: demanding a perfect passing bay or nothing
        # returned None for every candidate in the stack and froze the fleet.
        # Backing off even one cell breaks the symmetry of a mutual hard stop,
        # which is the thing that actually has to happen.
        return route if len(route) >= MIN_BACKOUT_CELLS else None

    def start_backout(self, now: float, mark: bool = True) -> bool:
        """
        Begin reversing, and REMEMBER WHY.

        The marking is not incidental. `finish_backout` replans from the new
        cell, and with nothing marked A* re-derives the route it was just
        pushed off -- so the robot drives forward into the same blockage and
        stalls again. Measured: of 50 completed backouts, 38 (76%) re-stalled
        within 3 seconds. Two of the three callers marked the blockage before
        retreating and the deadlock path did not, so it is done here once, for
        all of them.

        The belief decays at 0.05/s, so this is a 20 s hint rather than a
        permanent no-go: the corridor reopens on its own once it clears.
        """
        route = self.backout_target(now)
        if not route:
            return False
        if mark:
            if self.path_idx < len(self.path):
                self.table.mark_blocked(*self.path[self.path_idx], now,
                                        OBSTACLE_CONF)
            st = self.peers.get(self.waitfor_target)
            if st is not None:
                self.table.mark_blocked(*self.wmap.to_cell(st.x, st.y), now,
                                        OBSTACLE_CONF)
        self.saved_goal = self.goal
        self.path, self.path_idx = route, 0
        self.backing_out = True
        self.backout_deadline = now + T_BACKOUT
        self.stall_since = None
        self.backouts += 1
        return True

    def finish_backout(self, now: float) -> None:
        """Resume the real goal once clear, or on timeout."""
        self.backing_out = False
        self.backouts_done += 1
        self.goal = self.saved_goal
        self.saved_goal = None
        self.blocked_since = 0.0
        self.stall_since = None
        # Do NOT wipe the trail. It used to reset to [self.cell] here, which
        # destroys the only escape route the robot has: a robot that finishes
        # one backout and re-jams a second later then has trail_len == 1,
        # `backout_target` returns None forever, and it can never yield again.
        # Measured: R2 frozen 300 s with a one-cell trail. The trail is capped
        # at 60 cells anyway, so keeping it costs nothing.
        if self.goal:
            self.replan(now)

    def nearest_wide_cell(self, now: float,
                          max_r: int = 14) -> tuple[int, int] | None:
        """
        Closest reachable cell where two AMRs can pass, by breadth-first
        search outward from here.

        A yielder must end up somewhere the other robot can get past it.
        Retreating to another narrow cell just moves the deadlock.
        """
        from collections import deque
        start = self.cell
        seen = {start}
        q = deque([(start, 0)])
        while q:
            (cx, cy), d = q.popleft()
            if d and not self.wmap.is_narrow(cx, cy):
                return (cx, cy)
            if d >= max_r:
                continue
            for nb in self.wmap.neighbors(cx, cy):
                if nb in seen:
                    continue
                # never retreat INTO a peer
                wx, wy = self.wmap.to_world(*nb)
                if any(math.hypot(st.x - wx, st.y - wy) < R_HARD
                       for pid, st in self.peers.items()
                       if now - self.peer_seen_at.get(pid, -1e9) < 2.0):
                    continue
                seen.add(nb)
                q.append((nb, d + 1))
        return None

    # -- battery and charging bays ----------------------------------------

    def chargers(self) -> list[tuple[int, int]]:
        return [(n.cx, n.cy) for n in self.wmap.nodes.values()
                if n.kind == "charger"]

    def bay_taken(self, bay: tuple[int, int], now: float) -> bool:
        """
        Is a peer sitting on this bay, or heading to it?

        Decided from BROADCAST state only -- peer position from RobotState and
        peer destination from the last cell of their Intent path. No robot
        reads another's variables, so "nearest VACANT bay" stays decentralised
        and degrades honestly when comms are patchy.
        """
        bx, by = self.wmap.to_world(*bay)
        for pid, st in self.peers.items():
            if now - self.peer_seen_at.get(pid, -1e9) > 2.0:
                continue
            if math.hypot(st.x - bx, st.y - by) < 1.2:
                return True
        for pid, it in self.peer_intents.items():
            if now - self.peer_seen_at.get(pid, -1e9) > 2.0:
                continue
            if it.path_cells and tuple(it.path_cells[-1]) == bay:
                if pid < self.id:        # deterministic tie-break, both agree
                    return True
        # EXPLICIT CLAIM. The two tests above only say where a peer IS or
        # where its current path happens to END, so two robots could pick the
        # same bay from opposite sides of the map and neither would find out
        # until one arrived. A broadcast claim is visible the moment it is
        # made, which is what lets the loser divert while still in transit.
        # The EARLIER claim wins, so the outcome does not depend on who heard
        # whom first.
        owner = self.bay_claims.get(bay)
        if owner is not None and now - self.peer_seen_at.get(owner[0], -1e9) <= 5.0:
            mine = self.bay_claim_stamp if self.my_bay_claim == bay else None
            if mine is None or owner[1] < mine - 1e-9:
                return True
            if abs(owner[1] - mine) <= 1e-9 and owner[0] < self.id:
                return True
        return False

    def nearest_free_bay(self, now: float) -> tuple[int, int] | None:
        """Closest bay no peer has claimed. Falls back to closest if all taken."""
        cx, cy = self.cell
        bays = sorted(self.chargers(),
                      key=lambda b: abs(b[0] - cx) + abs(b[1] - cy))
        for b in bays:
            if not self.bay_taken(b, now):
                return b
        return bays[0] if bays else None

    def claim_bay(self, bay: tuple[int, int], now: float) -> None:
        """Announce that this bay is mine, with the time I decided it."""
        if self.my_bay_claim == bay and now - self.last_bay_bc < 1.0:
            return
        if self.my_bay_claim != bay:
            self.release_bay(now)
            self.bay_claim_stamp = now
        self.my_bay_claim = bay
        self.last_bay_bc = now
        eta = now + self.remaining_distance() / max(0.05, self.v_nom)
        bc = BayClaim(header=make_header(self.id, now),
                      bay_cx=bay[0], bay_cy=bay[1],
                      claimed_at=self.bay_claim_stamp, eta=eta)
        self.comms.send(wrap(bc, MsgType.BAY_CLAIM, self.id, BROADCAST, now),
                        now)

    def release_bay(self, now: float) -> None:
        if self.my_bay_claim is None:
            return
        bay, self.my_bay_claim = self.my_bay_claim, None
        bc = BayClaim(header=make_header(self.id, now),
                      bay_cx=bay[0], bay_cy=bay[1],
                      claimed_at=self.bay_claim_stamp, releasing=True)
        self.comms.send(wrap(bc, MsgType.BAY_CLAIM, self.id, BROADCAST, now),
                        now)

    def recheck_bay(self, now: float) -> bool:
        """
        En route to a bay someone else claimed first -> divert to a free one.

        Only WHILE IN TRANSIT. A robot already parked on a bay does not move
        off it: at that point the claim is settled by occupancy and shuffling
        parked robots costs aisle time for nothing.
        """
        if self.my_bay_claim is None or self.at_bay():
            return False
        if not self.bay_taken(self.my_bay_claim, now):
            return False
        alt = self.nearest_free_bay(now)
        if alt is None or alt == self.my_bay_claim:
            return False
        self.claim_bay(alt, now)
        self.goal = alt
        self.bay_diversions += 1
        return self.replan(now)

    def needs_charge(self) -> bool:
        """Low enough that charging outranks work. NOT 'am I idle'."""
        return self.battery < SOC_LOW

    def at_bay(self) -> bool:
        return self.cell in self.chargers()

    def go_idle(self, now: float) -> None:
        """
        No work: clear the aisles and sit on the nearest vacant bay.

        Parking on a bay is not the same as needing a charge. An idle robot
        left standing in an aisle blocks every peer routed through it, so it
        retires to a bay and tops up opportunistically while it waits.
        """
        bay = self.nearest_free_bay(now)
        if bay is None:
            return
        self.home_bay = bay
        self.goal = bay
        self.mode = RobotMode.IDLE
        self.claim_bay(bay, now)
        self.replan(now)

    def update_battery(self, dt: float, now: float) -> None:
        """Charge on a bay; stop charging once topped up."""
        if self.at_bay() and self.v < 0.05:
            if self.battery < 1.0:
                self.battery = min(1.0, self.battery + CHARGE_RATE * dt)
                self.charging = True
                self.mode = RobotMode.CHARGING
            if self.charging and self.battery >= SOC_RESUME:
                self.charging = False
                if self.task is None:
                    self.mode = RobotMode.IDLE
        else:
            self.charging = False

    def evaluate_task(self, task: Task, now: float) -> tuple[float, bool]:
        # Refuse work below SOC_LOW, and while topping up refuse until
        # SOC_RESUME -- otherwise a robot leaves the bay at 26%, takes a job,
        # and strands itself mid-aisle.
        if self.task is not None or self.battery < SOC_LOW:
            return INF_COST, False
        if self.charging and self.battery < SOC_RESUME:
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
        self.leg_start_dist = self.dist
        self.release_bay(now)      # working now; the bay is free for a peer
        self.phase = "to_pickup"
        self.mode = RobotMode.TO_PICKUP
        self.goal = (task.pickup_cx, task.pickup_cy)
        self.replan(now)

    def replan(self, now: float) -> bool:
        """
        Replan, keeping the SCHEDULE as well as the route.

        `plan` alone throws away the timing, and the timing is the entire
        point of a space-time search: a plan that says "hold at this cell for
        3 s and then go straight through" is indistinguishable, once
        collapsed to cells, from one that says "go straight through now".
        Executing it as the latter is how the planner's waits became
        invisible detour-or-nothing decisions.
        """
        if self.goal is None:
            return False
        cx, cy = self.cell
        st = plan_st(self.wmap, (cx, cy), self.goal, self.table, self.id,
                     now, now, self.v_nom, V_MAX, self.flags["congestion"])
        if not st:
            return False
        depart = schedule_of(st)
        path: list[tuple[int, int]] = []
        for (px, py, _t) in st:
            if not path or path[-1] != (px, py):
                path.append((px, py))
        # A hold is only recorded where the plan genuinely waits -- where the
        # scheduled departure is later than free-running arrival. Enforcing
        # the whole schedule would pin every robot to v_nom for its entire
        # route and quietly delete speed adaptation.
        holds = [0.0] * len(path)
        t = now
        step = CELL_SIZE / max(0.05, self.v_nom)
        for i in range(len(path)):
            if depart[i] > t + 1e-6:
                holds[i] = depart[i]
                self.planned_waits += 1
            t = max(depart[i], t) + step
        self.path, self.path_idx = path, 0
        self.path_holds = holds
        self.replans += 1
        return True

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

        # Direction along whichever axis we are actually moving on. The old
        # rule only ever looked at dy, so head-on meetings in a HORIZONTAL
        # aisle were invisible to it and never prevented.
        my_dx = nxt_cell[0] - self.cell[0]
        my_dy = nxt_cell[1] - self.cell[1]
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
            their_dx = their_dy = 0
            if math.hypot(st.vx, st.vy) > 0.05:
                their_dx = (st.vx > 0) - (st.vx < 0)
                their_dy = (st.vy > 0) - (st.vy < 0)
            else:
                it = self.peer_intents.get(rid)
                if it:
                    for c in it.path_cells[:6]:
                        if (c[0], c[1]) != (pc[0], pc[1]):
                            their_dx = (c[0] > pc[0]) - (c[0] < pc[0])
                            their_dy = (c[1] > pc[1]) - (c[1] < pc[1])
                            break

            # opposed on either axis -> head-on, do not enter
            if my_dx * their_dx + my_dy * their_dy < 0:
                return True

        # Capacity limit: a 3-cell-wide corridor cannot absorb a queue.
        # Holding at the mouth is cheap; gridlock inside is not.
        return occupants >= 2

    # -- decisions from LOCAL SENSING (no comms) ---------------------------

    def current_region(self) -> int:
        """
        Which contested region am I in? -1 = none.

        A region is an aisle segment. A robot sitting in the MOUTH of an
        aisle counts as contesting that aisle even though the mouth cell is
        open: the wedge that motivated all of this had two of its four robots
        standing in mouth cells, and a region test that excluded them gave
        those two no entry stamp and therefore no place in the stack.
        """
        seg = self.wmap.aisle_at(*self.cell)
        if seg >= 0:
            return seg
        for nb in self.wmap.neighbors(*self.cell):
            s2 = self.wmap.aisle_at(*nb)
            if s2 >= 0:
                return s2
        return -1

    def update_region(self, now: float) -> None:
        """Push on entering a region; the stamp is what LIFO pops on."""
        reg = self.current_region()
        if reg != self.region:
            self.region = reg
            self.region_entered_at = now if reg >= 0 else 0.0

    def note_trail(self) -> None:
        cell = self.cell
        if self.trail and self.trail[-1] == cell:
            return
        # Reversing over my own trail must not leave a loop behind me, or the
        # backout walks in circles. If I have stepped back onto a cell I
        # already occupied, truncate to it.
        if cell in self.trail:
            self.trail = self.trail[:self.trail.index(cell) + 1]
        else:
            self.trail.append(cell)
        del self.trail[:-60]

    def approaching_blind_corner(self, now: float) -> bool:
        """
        Should I drive at sight-limited speed right now?

        SENSOR AND RADIO TOGETHER. Geometry alone says every aisle mouth is
        dangerous, which on this map is 14% of the cells but 58% of the
        driving time -- slowing for all of them costs more throughput than
        the collisions it prevents. Geometry says WHERE a surprise is
        possible; the radio says whether anyone is in a position to supply
        one; the LiDAR overrides the radio when it can already see the peer
        the radio is warning about, because a peer in view is not hidden.

        The degraded case is the important one: with no fresh peer reports at
        all, nothing can be ruled out, so the rule reverts to slowing at every
        mouth. Losing the radio costs speed, never safety.
        """
        if self.path_idx >= len(self.path):
            return False
        n = max(1, int(D_BLIND_LOOKAHEAD / CELL_SIZE))
        ahead = [self.cell] + self.path[self.path_idx:self.path_idx + n]
        corners = [c for c in ahead if self.wmap.is_blind_corner(*c)]
        if not corners:
            return False

        fresh = [(pid, st) for pid, st in self.peers.items()
                 if now - self.peer_seen_at.get(pid, -1e9) <= 2.0]
        if not fresh:
            return True                     # blackout: cannot rule anyone out

        for (cx, cy) in corners:
            wx, wy = self.wmap.to_world(cx, cy)
            for pid, st in fresh:
                # Where could this peer be by the time I am at the corner?
                # Straight-line reachability at V_MAX is deliberately crude:
                # it over-estimates, and over-estimating risk is the safe
                # direction for a rule that can only slow me down.
                d = math.hypot(st.x - wx, st.y - wy)
                t_me = math.hypot(wx - self.x, wy - self.y) / V_MAX
                if d > V_MAX * (t_me + T_BLIND_REACT):
                    continue
                # If I can already see it, the corner is not hiding it.
                if any(math.hypot(tr.x - st.x, tr.y - st.y) < 0.6
                       for tr in self.lidar.tracks if tr.confirmed(now)):
                    continue
                return True
        return False

    def lidar_scan(self, now: float) -> None:
        """
        Turn stationary LiDAR tracks on my own path into routing knowledge.

        This is the comms-free half of obstacle handling. `announce_obstacle`
        needs the wedged robot to still have a working radio and needs me to
        receive it; this needs neither. A dead robot, a dropped pallet or a
        person standing in the aisle all look the same to the sensor, and all
        three should make me reroute.

        Deliberately NARROW: a track is only written into the reservation
        table if it is in my lane, ahead of me, stationary for T_STATIC, AND
        sitting on the path I still intend to drive. Marking every stationary
        return would blacklist every robot parked on a charging bay and
        poison the map for the whole fleet.
        """
        if not self.path or self.path_idx >= len(self.path):
            return
        blockers = self.lidar.blocking_tracks(self.x, self.y, self.theta,
                                              LANE_HALF_W, LIDAR_RANGE, now)
        if not blockers:
            return
        ahead = set(self.path[self.path_idx:])
        hit = False
        for t in blockers:
            cell = self.wmap.to_cell(t.x, t.y)
            if cell not in ahead:
                continue
            self.table.mark_blocked(*cell, now, LIDAR_BLOCK_CONF)
            self.lidar_blocks += 1
            hit = True
        if not hit or self.goal is None:
            return
        # Rate-limit: one sensor tick must not trigger a replan storm. The
        # blockage belief persists in the table, so a suppressed replan is
        # not a lost one -- the next scheduled replan still sees it.
        if now - self.last_lidar_replan < T_LIDAR_REPLAN:
            return
        self.last_lidar_replan = now
        if self.replan(now):
            self.lidar_replans += 1

    def lidar_speed_cap(self, now: float) -> float:
        """
        Speed cap derived from tracks alone. Can only ever REDUCE speed.

        Resolved by whichever mechanism the arm under test uses: the full
        system computes a slowdown, the stop-and-wait baseline halts. Putting
        speed adaptation on both arms here would leak the mechanism under
        test into its own control.
        """
        hit = self.lidar.closing_track(self.x, self.y, self.theta,
                                       LANE_HALF_W, LIDAR_RANGE, now)
        if hit is None:
            return V_MAX
        track, ttc = hit
        if ttc > 6.0:
            return V_MAX                      # far enough to ignore
        self.lidar_decisions += 1
        if not self.flags["speed_adapt"]:
            if self.v > 0.01:
                self.stops += 1
            return 0.0
        gap = max(0.05, math.hypot(track.x - self.x, track.y - self.y)
                  - D_COLLIDE)
        v_new, action = adapt_speed(gap, now + ttc + 1.0, now, self.v,
                                    V_MIN, V_MAX)
        if action == "STOP" and self.v > 0.01:
            self.stops += 1
        return v_new

    def segment_entry_deferred(self, now: float) -> bool:
        """
        PREDICTIVE. Refuse to enter an aisle I am forecast to jam.

        `corridor_blocked` only ever asked whether a peer is inside the aisle
        RIGHT NOW. That is too late: two robots converging on opposite mouths
        both see an empty corridor, both enter, and no velocity solution
        exists once they are in. Every wedge measured on this map formed that
        way.

        Here each robot derives, from the `Intent` path cells its peers
        already broadcast, WHEN each of them will be inside each segment and
        WHICH WAY -- so an opposed overlap is visible while both robots are
        still outside and either can still cheaply stop.

        Who defers is decided the same way the stack pops: whoever would
        enter LATER waits. First come, first served; the later arrival holds
        at the mouth for a few seconds instead of buying a deadlock. Both
        robots compute the identical comparison from the identical broadcast
        data, so they cannot both defer (which would livelock) and cannot
        both proceed. Simultaneous entries inside SEG_TIE are split by
        priority, which is a total order.

        No new message type: this is the Intent that was already on the air.
        """
        if self.path_idx >= len(self.path) or self.backing_out:
            return False
        nxt = self.path[self.path_idx]
        seg = self.wmap.aisle_at(*nxt)
        if seg < 0 or seg == self.wmap.aisle_at(*self.cell):
            return False                      # not entering a new segment

        mine = segment_windows(self.path[self.path_idx:], now, self.v_nom,
                               self.wmap)
        theirs = {}
        for pid, it in self.peer_intents.items():
            if now - self.peer_seen_at.get(pid, -1e9) > 2.0:
                continue
            theirs[pid] = segment_windows(
                [tuple(c) for c in it.path_cells], now,
                max(0.05, it.nominal_speed), self.wmap)
        hit = predicted_head_on(mine, theirs)
        model = self.flags.get("risk")
        if hit is None:
            # The rule sees nothing. Ask the model whether it sees something
            # the rule's opposed-heading test cannot: that test needs a peer
            # to be MOVING towards me, and in a forming jam everyone has
            # already slowed down. The hand-written baseline scores recall
            # 0.001 for exactly this reason.
            if model is not None:
                p = model.risk(deadlock_features(self, now))
                if p >= model.HIGH_CONF:
                    self.risk_deferrals += 1
                    self.entry_deferrals += 1
                    return True
            return False
        seg_id, pid, my_entry = hit
        their_entry = min((w[1] for w in theirs[pid] if w[0] == seg_id),
                          default=1e9)
        if my_entry < their_entry - SEG_TIE:
            # I get there first -- unless the model is confident this is a
            # jam. Overriding only in the CAUTIOUS direction is what keeps
            # this safe: a wrong model costs a wait, never a collision.
            if model is not None:
                p = model.risk(deadlock_features(self, now))
                if p >= model.HIGH_CONF:
                    self.risk_deferrals += 1
                    self.entry_deferrals += 1
                    return True
            return False
        if abs(my_entry - their_entry) <= SEG_TIE:
            peer_p = self.peer_intents[pid].priority or [1e9]
            if tuple(self.priority(now)) < tuple(peer_p):
                return False                  # dead heat, I have right of way
        self.entry_deferrals += 1
        return True

    def decide_speed(self, now: float) -> float:
        """
        Speed adaptation (doc E.6). This is the mechanism that beats
        stop-and-wait: compute an arrival time that clears the conflict and
        derive the speed for it, rather than halting.
        """
        # corridor reservation applies to BOTH arms (it is a safety rule,
        # not the mechanism under test)
        # Both evaluated, deliberately not short-circuited: an `or` would
        # leave the predictive rule uncounted whenever the reactive one fired
        # first, and an uncounted rule is indistinguishable from a dead one.
        reactive = self.corridor_blocked(now)
        predictive = self.segment_entry_deferred(now)
        if reactive or predictive:
            return 0.0

        # Execute a PLANNED wait as a slowdown, not as a halt. The planner
        # decided this cell should be left at a particular time; the speed to
        # achieve that is a computed quantity, and it falls to zero only when
        # it drops below v_min -- the same rule as every other stop here.
        sched_cap = V_MAX
        if 0 < self.path_idx <= len(self.path_holds):
            hold = self.path_holds[self.path_idx - 1]
            if hold > now:
                self.waits_executed += 1
                d = self.dist_to_waypoint()
                v_req, act = adapt_speed(d, hold, now, self.v, V_MIN, V_MAX)
                sched_cap = v_req
                if act == "STOP" and self.v > 0.01:
                    self.stops += 1

        # Drive within sight distance. Geometry, not comms -- it holds in a
        # blackout, and it applies to BOTH arms because it is a sensing limit
        # rather than the conflict-resolution mechanism under test.
        blind_cap = V_MAX
        if self.approaching_blind_corner(now):
            blind_cap = V_BLIND
            self.blind_slowdowns += 1

        # LOCAL SENSING ARM. Runs first and unconditionally: it must still
        # produce a decision when every packet is being dropped, which is the
        # whole reason it exists. It can only lower the final command.
        lidar_cap = min(self.lidar_speed_cap(now), blind_cap, sched_cap)

        if not self.flags["speed_adapt"]:
            return min(self._stop_and_wait_speed(now), lidar_cap)

        mine = sample_trajectory(self.path[self.path_idx:], now, self.v_nom)
        if not mine:
            return min(self.v_nom, lidar_cap)
        others = {}
        for rid, it in self.peer_intents.items():
            if now - self.peer_seen_at.get(rid, 0) > 2.0:
                continue
            others[rid] = sample_trajectory(
                [tuple(c) for c in it.path_cells], now,
                max(0.05, it.nominal_speed))

        hit = first_conflict(mine, others)
        if hit is None:
            return min(V_MAX, lidar_cap)

        t_c, pid, _, _ = hit
        my_p = self.priority(now)
        peer_p = self.peer_intents[pid].priority or [1e9]
        if tuple(my_p) < tuple(peer_p):
            return min(V_MAX, lidar_cap)    # right of way, still sensor-capped

        dist = max(0.1, (t_c - now) * self.v_nom)
        peer_exit = t_c + 1.0
        v_new, action = adapt_speed(dist, peer_exit, now, self.v,
                                    V_MIN, V_MAX)
        if action == "STOP" and self.v > 0.01:
            self.stops += 1          # count TRANSITIONS into stop, not ticks
        return min(v_new, lidar_cap)

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
                          sensed: list[tuple[float, float]],
                          reverse: bool = False) -> float:
        """
        L2. Deterministic geometric brake. FINAL AUTHORITY over speed.

        Two properties that matter:
          * It can only ever REDUCE v_cmd, never raise it. No learned
            component can talk its way past this.
          * It runs on LiDAR returns, NOT on comms. So it keeps working
            when the network is dead -- which is exactly why the fleet
            stays collision-free in degraded mode.
        """
        v_out = v_cmd
        # Travel direction, which is NOT the same as heading when reversing.
        # The brake must watch the end of the robot that is moving first --
        # checking the front lane while backing up is how you reverse into
        # somebody at full speed.
        sgn = -1.0 if reverse else 1.0
        hx = math.cos(self.theta) * sgn
        hy = math.sin(self.theta) * sgn
        half_w = LANE_HALF_W_REVERSE if reverse else LANE_HALF_W
        if reverse:
            in_lane_now = self.wmap.is_narrow(*self.cell)
            v_out = min(v_out, V_REVERSE_LANE if in_lane_now else V_REVERSE)
        for (px, py) in sensed:
            dx, dy = px - self.x, py - self.y
            d = math.hypot(dx, dy)

            # Omnidirectional floor: genuine body contact, whatever the bearing.
            if d < D_COLLIDE:
                return 0.0

            fwd = dx * hx + dy * hy           # along my heading
            lat = -dx * hy + dy * hx          # across my heading
            if fwd <= 0.0:
                continue                      # behind me: not my problem
            if abs(lat) >= half_w:
                continue                      # will clear me sideways

            if fwd < R_HARD:
                return 0.0                    # in my lane, too close
            if fwd < R_SLOW:
                scale = (fwd - R_HARD) / (R_SLOW - R_HARD)
                v_out = min(v_out, v_cmd * scale)
        return max(0.0, v_out)

    def step(self, now: float, dt: float,
             sensed: list[tuple[float, float]] | None = None) -> None:
        # Sensing updates even when parked: a robot sitting on a bay still
        # needs to know what is around it, and its tracks must not go stale
        # and then jump when it sets off again.
        self.lidar.update(sensed or [], now, dt)
        self.update_region(now)
        self.note_trail()
        if not self.path or self.path_idx >= len(self.path):
            self.v = 0.0
            return

        self.lidar_scan(now)
        v_cmd = self.decide_speed(now)
        # L2 has the last word, always.
        self.v = self.safety_supervisor(v_cmd, sensed or [], self.backing_out)
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

        # REVERSING. A backout drives the trail backwards without turning
        # round, because a 0.98 m body cannot turn in a 1.50 m aisle and
        # swinging it would sweep through the very peer it is escaping. Real
        # AMRs reverse out of exactly this situation, and holding the heading
        # is also what frees the robot: the peer that hard-stopped it is now
        # BEHIND the direction of travel, so the front lane is clear and the
        # supervisor stops vetoing the move.
        if not self.backing_out:
            self.theta = math.atan2(ny - self.y, nx - self.x)
        self.dist += math.hypot(nx - self.x, ny - self.y)
        self.x, self.y = nx, ny
        self.battery = max(0.0, self.battery - move * 0.0008)


class Simulation:
    def __init__(self, n_robots: int = 4, seed: int = 0,
                 congestion: bool = True, speed_adapt: bool = True,
                 loss_rate: float = 0.0, use_policy: bool = False,
                 use_risk_model: bool = False, lifo_stack: bool = True):
        self.rng = random.Random(seed)
        self.wmap = WarehouseMap()
        ids = list(range(1, n_robots + 1))
        self.comms = CommsMediator(ids, random.Random(seed),
                                   loss_rate=loss_rate)
        # `use_policy` selects the ACTION policy and is still unread --
        # DEAD_CODE.md Tier 3, unchanged by this work and still honest.
        # `use_risk_model` is the deadlock-risk predictor and IS consumed,
        # in Robot.segment_entry_deferred.
        risk = None
        if use_risk_model and DeadlockRiskModel.available():
            risk = DeadlockRiskModel()
        flags = {"congestion": congestion, "speed_adapt": speed_adapt,
                 "policy": use_policy, "risk": risk, "lifo": lifo_stack}

        # AMRs begin on charging bays, not scattered mid-aisle. A robot that
        # starts stray is already obstructing a corridor at t=0, which biases
        # every congestion measurement from the first tick.
        bays = [(n.cx, n.cy) for n in sorted(self.wmap.nodes.values(),
                                             key=lambda n: n.name)
                if n.kind == "charger"]
        if n_robots > len(bays):
            raise ValueError(f"{n_robots} robots but only {len(bays)} bays")
        starts = bays[:n_robots]
        self.robots = [Robot(i, *starts[i - 1], self.wmap, self.comms, flags)
                       for i in ids]
        for r, b in zip(self.robots, starts):
            r.home_bay = b

        self.t = 0.0
        self.metrics = Metrics()
        self.tasks: list[Task] = []
        self.next_task_id = 1
        self.open_tasks: dict[int, Task] = {}
        self.task_announced_at: dict[int, float] = {}
        # contact latches, so a sustained graze counts once (see BUG 7)
        self._contact: dict[tuple[int, int], bool] = {}
        self._near: dict[tuple[int, int], bool] = {}

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
        """
        Count contact EVENTS, not ticks.

        ENGINEERING_LOG BUG 7 fixed exactly this defect in the stop metric and
        left it in place here. One sustained graze between a moving robot and
        a parked one reported 1359 "collisions" on seed 5 -- it was a single
        contact lasting 136 s. Tick counting makes a long mild contact look
        catastrophic and two brief hard ones look benign, so it cannot be
        compared across runs at all.

        `collision_ticks` is kept alongside because DURATION of contact is
        real information; it is just not a count of collisions.
        """
        for i, a in enumerate(self.robots):
            for b in self.robots[i + 1:]:
                key = (a.id, b.id)
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < D_COLLIDE:
                    self.metrics.collision_ticks += 1
                    if not self._contact.get(key):
                        self.metrics.collisions += 1     # transition INTO contact
                    self._contact[key] = True
                else:
                    if self._contact.get(key):
                        self._contact[key] = False
                    if d < D_NEAR:
                        self.metrics.near_miss_ticks += 1
                        if not self._near.get(key):
                            self.metrics.near_misses += 1
                        self._near[key] = True
                    else:
                        self._near[key] = False

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
            # A stalled robot announces itself as an obstacle and publishes
            # its wait-for edge; peers reroute and elect a yielder locally.
            r.announce_obstacle(self.t)
            r.announce_waitfor(self.t)
            r.announce_pushback(self.t)
            r.recheck_bay(self.t)
            # Order matters: a real cycle is the LIFO stack's business, a
            # negotiated verdict beats a guess, and the unilateral reverse is
            # the last resort for a stall nobody has claimed responsibility
            # for.
            if r.check_deadlock(self.t):
                self.metrics.deadlocks += 1
            elif r.check_pushback(self.t):
                self.metrics.pushbacks += 1
            elif r.check_stuck_reverse(self.t):
                self.metrics.stuck_reverses += 1
            r.update_battery(DT, self.t)
            # Low battery outranks everything: hand the task back and head
            # for a bay. Decided by the robot from its own SoC.
            if r.needs_charge() and not r.charging:
                if r.task is not None and r.phase == "to_pickup":
                    r.release_task(r.task.task_id, self.t,
                                   ReleaseReason.BATTERY)
                if r.task is None and not r.at_bay():
                    bay = r.nearest_free_bay(self.t)
                    if bay and r.goal != bay:
                        r.goal = bay
                        r.mode = RobotMode.CHARGING
                        r.claim_bay(bay, self.t)
                        r.replan(self.t)
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
            # Range AND line of sight. Without the raycast a robot detected
            # peers straight through solid racking, so its braking was better
            # than any real sensor could be -- yet another shortcut that
            # biased the result favourably.
            sensed = []
            for o in self.robots:
                if o.id == r.id:
                    continue
                d = math.hypot(o.x - r.x, o.y - r.y)
                if d >= LIDAR_RANGE:
                    continue
                # Occlusion is a LONG-RANGE optical property. At CONTACT
                # range it is not physical: a rack corner cannot hide a 0.98 m
                # wide body whose centre is 0.97 m from yours -- the two
                # footprints already overlap, and a real AMR's bumper does not
                # care what the scanner can see. Without the exemption two
                # robots grazed at 0.974 m with the supervisor never firing
                # (1359 contact ticks, seed 5).
                #
                # The threshold is D_NEAR: the swept circles of two 0.70 m
                # radius bodies touch there, so anything closer is a genuine
                # proximity event that a bumper ring would register.
                #
                # This DOES create mutual hard stops -- an occluded peer
                # between R_HARD (1.10 m) and D_NEAR triggers a brake that
                # the supervisor cannot lift, because it may only reduce
                # speed. Narrowing the threshold to D_COLLIDE to dodge that
                # was the wrong trade: it bought 0 deadlock-freezes at the
                # price of 3 collisions over 10 seeds, and a collision is not
                # a currency this project spends.
                #
                # Breaking a mutual stop is the DEADLOCK layer's job, not the
                # brake's, and it can now actually do it: cycles are detected
                # (the wait-for edge names whoever the brake is stopping for),
                # the election falls through to a candidate that can move, and
                # the yielder REVERSES out without turning round.
                if d >= D_NEAR and not self.wmap.line_of_sight(
                        r.x, r.y, o.x, o.y):
                    self.metrics.lidar_occluded += 1
                    continue
                sensed.append((o.x, o.y))
            r.step(self.t, DT, sensed)

            # Backout complete -> resume the real goal. Recovery time is
            # measured from the moment the stack was popped, not per tick:
            # the old code appended DT to deadlock_recovery_s, which recorded
            # the sampling interval rather than how long recovery took.
            if r.backing_out and (r.path_idx >= len(r.path)
                                  or self.t > r.backout_deadline):
                self.metrics.deadlock_recovery_s.append(
                    self.t - (r.backout_deadline - T_BACKOUT))
                r.finish_backout(self.t)
                continue

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
                    r.leg_start_dist = r.dist
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
                    # Clear the aisle: retire to the nearest bay no peer has
                    # claimed. The robot picks this itself from broadcast state.
                    r.go_idle(self.t)

        self.check_collisions()
        self.aggregate_metrics()

    def aggregate_metrics(self) -> None:
        """
        Roll per-robot counters up into Metrics.

        Called from `step`, NOT only from `run`. It used to live at the end of
        `run` alone, which meant every per-robot counter read ZERO for any
        consumer that drives `step` directly -- and the dashboard is exactly
        such a consumer. The live UI reported `deadlocks 4, backouts 0/0`,
        which is not a state this code can reach: `deadlocks` only increments
        when a backout starts. That impossible pair is what exposed it.

        `replans`, `full_stops`, `time_stopped` and `distance_m` had the same
        defect before any of the new counters existed, so the dashboard has
        been showing zeros for those since it was written.

        Cost is a dozen sums over 4 robots per 0.1 s tick. Immeasurable next
        to the planner, and the alternative is a metric that is silently dead
        in the one view a human actually looks at.
        """
        m, rs = self.metrics, self.robots
        m.sim_time = self.t
        m.full_stops = sum(r.stops for r in rs)
        m.time_stopped = sum(r.stopped_time for r in rs)
        m.distance = sum(r.dist for r in rs)
        m.replans = sum(r.replans for r in rs)
        m.lidar_blocks = sum(r.lidar_blocks for r in rs)
        m.lidar_decisions = sum(r.lidar_decisions for r in rs)
        m.lidar_replans = sum(r.lidar_replans for r in rs)
        m.blind_slowdowns = sum(r.blind_slowdowns for r in rs)
        m.backouts = sum(r.backouts for r in rs)
        m.backouts_done = sum(r.backouts_done for r in rs)
        m.entry_deferrals = sum(r.entry_deferrals for r in rs)
        m.planned_waits = sum(r.planned_waits for r in rs)
        m.waits_executed = sum(r.waits_executed for r in rs)
        m.risk_deferrals = sum(r.risk_deferrals for r in rs)
        m.yield_skips = sum(r.yield_skips for r in rs)
        m.stuck_reverses = sum(r.stuck_reverses for r in rs)
        m.pushbacks_sent = sum(r.pushbacks_sent for r in rs)
        m.bay_diversions = sum(r.bay_diversions for r in rs)

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

        self.aggregate_metrics()
        out = self.metrics.summary()
        out["comms"] = self.comms.stats()
        return out


if __name__ == "__main__":
    sim = Simulation(n_robots=4, seed=1)
    import json
    print(json.dumps(sim.run(), indent=2))
