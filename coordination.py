"""
Coordination layer.

Contains the four mechanisms that make this system decentralised:

  1. consensus sealed-bid auction    (doc E.1, E.2)  -- no auctioneer
  2. conflict prediction             (doc E.4)
  3. SPEED ADAPTATION                (doc E.6)  <-- the headline mechanism
  4. deadlock cycle detection        (doc E.7)

Speed adaptation is the one that produces the throughput number. Stop-and-wait
halts a robot for the full duration of every conflict; here a stop is a
COMPUTED last resort, reached only when the required speed falls below v_min.
"""

from __future__ import annotations

import math

from amr_msgs import INF_COST, Reservation, priority_tuple
from warehouse_map import CELL_SIZE

# bid weights (doc E.2)
ALPHA, BETA, GAMMA, DELTA, EPSILON = 1.0, 0.5, 40.0, 15.0, 8.0

# auction timing, seconds (doc 0.3). Protocol timings, not calibration
# constants: they set how long a robot listens before it commits, and so
# bound how badly a lost Bid can desync the fleet.
T_BID = 0.3        # bid collection window
T_CLAIM = 0.4      # wait for the believed winner's Claim before re-running
T_REBID = 0.5      # retry cadence for a task nobody could take yet
T_RELEASE = 8.0    # stalled this long while fetching -> give the task back
T_COOLDOWN = 12.0  # ...and don't re-bid on it for this long

# battery policy. A robot does NOT visit a charger because it is idle --
# only because it is actually low. Idling ON a bay is different from
# charging: it parks there to clear the aisles, and tops up while it waits.
SOC_LOW = 0.25       # below this, charging outranks any task
SOC_RESUME = 0.90    # charge to here before bidding again
CHARGE_RATE = 0.02   # SoC per second on a bay

# A stalled robot IS an obstacle. Announce it, let peers route around it.
T_OBSTACLE = 3.0     # stalled this long -> broadcast myself as blocking
T_WAITFOR = 1.0      # cadence for wait-for edge broadcasts
OBSTACLE_CONF = 1.0  # initial belief; decays 0.05/s in ReservationTable

# conflict prediction (doc E.4)
HORIZON = 10.0
DT_SAMPLE = 0.5
TAU_C = 4.0
D_SAFE = 0.8

# speed adaptation (doc E.6)
V_MIN = 0.15
T_BUFFER = 1.0
HYSTERESIS = 0.08

# deadlock (doc E.7)
T_STALL = 3.0
V_STALL = 0.05

# LIFO deadlock resolution.
#
# The cycle member that entered the contested region LAST backs out. This is
# a stack: entering a corridor pushes, resolving pops, and it pops from the
# top. Two reasons, one practical and one structural.
#
# Practical: in a corridor the last robot in is by construction the one
# NEAREST the entrance, so its escape route is the stretch it has just driven
# and nobody is standing in it -- anyone behind it entered later still and has
# already been popped. Yielding by priority has no such guarantee: it can
# elect a robot buried at the far end of the aisle, whose only way out is
# through the very peer it is deadlocked with. That is exactly what was
# happening here, and it is why nine detected deadlocks produced no recovery.
#
# Structural: this is timestamp deadlock resolution -- Rosenkrantz, Stearns &
# Lewis (1978), the WAIT-DIE rule from distributed databases. Younger
# transaction aborts, older one proceeds. It is starvation-free for the same
# reason theirs is: a robot's entry stamp only gets older relative to new
# arrivals, so a robot cannot be chosen to yield forever.
T_BACKOUT = 20.0     # give up on a backout that has not cleared in this long
LIFO_EPS = 0.25      # entry stamps within this are a tie -> distance decides
# How long the fleet waits for an elected robot to confirm it is backing out
# before electing the next one down the stack. Election is not the same thing
# as ABILITY: the top of the stack may have a peer parked in its reverse path
# and no route out. Without this the election has one candidate and no
# fallback, so an elected robot that cannot move freezes the whole fleet --
# measured at 300 s of total paralysis. Confirmation is observable because
# WaitFor.backing_out is broadcast.
T_YIELD_CONFIRM = 3.0

# Negotiated push-back.
#
# When robot A is halted because robot B is in its lane, A asks B to move
# and BOTH evaluate the same rule, so the answer is agreed rather than
# guessed. Before this, each robot decided to reverse on its own evidence,
# which is symmetric: either both reversed, or the one that reversed drove
# straight back into the same blockage. Measured: robots ping-ponging between
# two cells for 773-1281 ticks per 4000-tick run, 20-32% of the time.
#
# Order: PRIORITY first, then DISTANCE COVERED on the current leg. The robot
# that has driven further keeps going and the one that has driven less
# reverses, because progress is sunk cost -- making the nearly-finished robot
# give way throws away the most work and it will only have to redo it. It is
# also the same principle as the LIFO stack expressed in metres instead of
# seconds: whoever has least invested in this corridor is cheapest to move.
PROGRESS_EPS = 0.5   # metres; closer than this is a tie, decide by id

# predictive segment reservation
SEG_HORIZON = 14.0   # s of announced path scanned for segment traversals


# --------------------------------------------------------------------------
# 1. Consensus sealed-bid auction
# --------------------------------------------------------------------------

def feasible(capacity_kg: float, payload_kg: float, task_kg: float,
             battery_soc: float, energy_needed: float,
             energy_full: float, margin: float = 0.10) -> bool:
    """Hard gate (doc E.1). Evaluated BEFORE bidding."""
    if task_kg > capacity_kg - payload_kg:
        return False
    available = battery_soc * energy_full
    if energy_needed + margin * energy_full > available:
        return False
    return True


def bid_cost(eta: float, congestion: float, energy_needed: float,
             energy_available: float, queue_len: int,
             conflict_risk: float) -> float:
    """
    Doc E.2. Lower wins.

    The energy term is a FRACTION of remaining charge, not an absolute.
    That single choice solves "5 m away at 8%% battery vs 15 m at 80%%"
    with no special-case rule: the same absolute energy is a large fraction
    for a depleted robot, so its bid inflates automatically.
    """
    if energy_available <= 1e-6:
        return INF_COST
    return (ALPHA * eta
            + BETA * congestion
            + GAMMA * (energy_needed / energy_available)
            + DELTA * queue_len
            + EPSILON * conflict_risk)


def has_quorum(n_bidders: int, fleet_size: int) -> bool:
    """
    May a robot COMMIT to a task on the evidence it has?

    A robot that cannot hear its peers hears only its own bid, so it always
    "wins" -- and in a blackout every robot wins the same task and four AMRs
    drive to one pallet. Requiring a majority of the fleet to have been heard
    turns that silent duplicate allocation into an honest refusal to commit.

    This is the standard no-quorum-no-commit rule. A lone robot (fleet of 1)
    is trivially its own majority.
    """
    if fleet_size <= 1:
        return True
    return n_bidders * 2 > fleet_size


def resolve_auction(bids: dict[int, tuple[float, bool]]) -> int:
    """
    Every robot runs THIS on the same bid set and gets the same answer.
    No auctioneer exists. Tie-break by lowest robot_id keeps it deterministic.

    bids: {robot_id: (cost, feasible)}
    """
    valid = [(c, rid) for rid, (c, f) in bids.items()
             if f and c < INF_COST]
    if not valid:
        return 0
    valid.sort(key=lambda t: (t[0], t[1]))    # cost, then id
    return valid[0][1]


# --------------------------------------------------------------------------
# 2. Conflict prediction
# --------------------------------------------------------------------------

def sample_trajectory(path: list[tuple[int, int]], t_start: float,
                      v_nom: float, horizon: float = HORIZON
                      ) -> list[tuple[float, float, float]]:
    """(t, x, y) samples along a path at nominal speed."""
    out = []
    if not path:
        return out
    step_time = CELL_SIZE / max(0.05, v_nom)
    t = t_start
    for (cx, cy) in path:
        if t - t_start > horizon:
            break
        out.append((t, (cx + 0.5) * CELL_SIZE, (cy + 0.5) * CELL_SIZE))
        t += step_time
    return out


def conflict_risk(my_traj, peer_trajs: dict[int, list],
                  d_safe: float = D_SAFE) -> float:
    """Doc E.4. Imminent conflicts weigh exponentially more than distant ones."""
    if not my_traj:
        return 0.0
    total = 0.0
    t0 = my_traj[0][0]
    for traj in peer_trajs.values():
        if not traj:
            continue
        for (t, x, y) in my_traj:
            px, py = _interp(traj, t)
            if px is None:
                continue
            if math.hypot(x - px, y - py) < d_safe:
                total += math.exp(-(t - t0) / TAU_C)
    return total


def _interp(traj, t):
    if not traj or t < traj[0][0] or t > traj[-1][0]:
        return None, None
    for i in range(len(traj) - 1):
        t0, x0, y0 = traj[i]
        t1, x1, y1 = traj[i + 1]
        if t0 <= t <= t1:
            a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    return traj[-1][1], traj[-1][2]


def first_conflict(my_traj, peer_trajs: dict[int, list],
                   d_safe: float = D_SAFE):
    """Earliest predicted conflict: (t, peer_id, x, y) or None."""
    best = None
    for pid, traj in peer_trajs.items():
        for (t, x, y) in my_traj:
            px, py = _interp(traj, t)
            if px is None:
                continue
            if math.hypot(x - px, y - py) < d_safe:
                if best is None or t < best[0]:
                    best = (t, pid, x, y)
                break
    return best


# --------------------------------------------------------------------------
# 3. SPEED ADAPTATION  --  the anti-stop-and-wait mechanism
# --------------------------------------------------------------------------

def adapt_speed(dist_to_conflict: float, peer_exit_time: float, now: float,
                v_cur: float, v_min: float = V_MIN, v_max: float = 0.8,
                t_buffer: float = T_BUFFER) -> tuple[float, str]:
    """
    Doc E.6. Returns (new_speed, action).

    Instead of stopping, compute the arrival time that clears the conflict
    and derive the speed that achieves it. Stopping happens ONLY when the
    required speed is below v_min -- i.e. it is a computed outcome, never
    a default reflex.
    """
    t_target = peer_exit_time + t_buffer
    dt = t_target - now
    if dt <= 1e-3:
        return v_max, "PROCEED"

    v_new = dist_to_conflict / dt
    if v_new >= v_max:
        return v_max, "PROCEED"
    if v_new >= v_min:
        if abs(v_new - v_cur) < HYSTERESIS:      # anti-chatter
            return v_cur, "SLOW"
        return v_new, "SLOW"
    return 0.0, "STOP"        # last resort only


def resolve_by_priority(my_prio: list[float], peer_prio: list[float]) -> bool:
    """True if I have right of way. Both robots compute this identically."""
    return tuple(my_prio) < tuple(peer_prio)


# --------------------------------------------------------------------------
# 4. Deadlock detection
# --------------------------------------------------------------------------

class DeadlockDetector:
    """
    Each robot builds the fleet-wide wait-for graph locally from broadcast
    WaitFor edges, then runs DFS cycle detection. At N=5 this is trivial.

    Resolution is DETERMINISTIC. A learned policy that resolves deadlocks
    *sometimes* is worse than a rule that resolves them *always*.
    """

    def __init__(self) -> None:
        self.edges: dict[int, int] = {}        # waiter -> blocker
        self.stall_since: dict[int, float] = {}

    def update_stall(self, robot_id: int, speed: float, now: float) -> bool:
        if speed < V_STALL:
            self.stall_since.setdefault(robot_id, now)
            return (now - self.stall_since[robot_id]) > T_STALL
        self.stall_since.pop(robot_id, None)
        return False

    def set_edge(self, waiter: int, blocker: int) -> None:
        if blocker == 0:
            self.edges.pop(waiter, None)
        else:
            self.edges[waiter] = blocker

    def find_cycle(self) -> list[int]:
        """Returns cycle members, or [] if none."""
        for start in list(self.edges):
            seen, node = [], start
            while node in self.edges:
                if node in seen:
                    return seen[seen.index(node):]
                seen.append(node)
                node = self.edges[node]
        return []

    def blocked_set(self, cycle: list[int]) -> list[int]:
        """
        The cycle PLUS everyone queued behind it -- the whole stack.

        This is the correction that made LIFO actually work. A cycle names the
        robots that are deadlocked, but not the robots whose bodies are in the
        way of resolving it. Measured on seed 3: the cycle was {2,4}, robot 2
        was correctly and unanimously elected to back out, and it failed to do
        so 1146 times running because robot 1 -- merely QUEUED behind robot 2,
        never part of the cycle -- was parked in its reverse path.

        Robot 1 had entered the aisle at t=68.2 against robot 2's t=48.8, so
        it was the true top of the stack and should have popped first. Electing
        from the cycle alone could never see that.

        So the resolution set is the cycle plus the transitive closure of
        wait-for edges leading INTO it. Popping its latest entrant frees the
        next one down, and the stack unwinds in reverse order of entry.
        """
        if not cycle:
            return []
        members = set(cycle)
        changed = True
        while changed:                      # walk the wait-for tree inward
            changed = False
            for waiter, blocker in self.edges.items():
                if blocker in members and waiter not in members:
                    members.add(waiter)
                    changed = True
        return sorted(members)

    @staticmethod
    def rank_yielders_lifo(cycle: list[int],
                           entered_at: dict[int, float],
                           dist_to_block: dict[int, float],
                           priorities: dict[int, list[float]]) -> list[int]:
        """
        The whole stack in pop order, latest entrant first.

        Returning the full ranking rather than just the winner is what makes a
        FALLBACK possible. Being elected and being able to move are different
        things: the top of the stack may have a peer standing in its reverse
        path. With a single-candidate election there is nobody to fall back
        to, and the fleet waits forever on a robot that cannot go anywhere.
        """
        if not cycle:
            return []
        return sorted(
            cycle,
            key=lambda r: (-entered_at.get(r, 0.0),
                           dist_to_block.get(r, 1e9),
                           [-v for v in priorities.get(r, [-1e9])],
                           -r))

    @staticmethod
    def choose_yielder_lifo(cycle: list[int],
                            entered_at: dict[int, float],
                            dist_to_block: dict[int, float],
                            priorities: dict[int, list[float]]) -> int:
        """
        LIFO: the LAST robot into the contested region backs out.

        Every robot in the cycle computes this from broadcast WaitFor fields
        and gets the same answer, so there is still no negotiation round-trip
        and no referee.

        Ordering, in full:
          1. latest `entered_at`  -- the top of the stack
          2. then SHORTEST distance to the blocking robot. Two robots that
             entered a corridor from opposite ends within LIFO_EPS have no
             meaningful stack order, so the tie goes to whoever is closest to
             the pinch point: it has the least corridor to clear and frees
             the jam soonest.
          3. then lowest priority, then highest id -- both total and
             deterministic, so the answer can never be ambiguous.

        A robot with no entry stamp (never entered a tracked region, or its
        WaitFor was lost) reports 0.0 and therefore sorts as the OLDEST
        entrant. Missing evidence must never volunteer a robot to yield --
        that would let a dropped packet elect a victim.
        """
        if not cycle:
            return 0
        top = max(entered_at.get(r, 0.0) for r in cycle)
        tied = [r for r in cycle
                if top - entered_at.get(r, 0.0) <= LIFO_EPS]
        return min(tied, key=lambda r: (dist_to_block.get(r, 1e9),
                                        [-v for v in
                                         priorities.get(r, [-1e9])],
                                        -r))

    @staticmethod
    def choose_yielder(cycle: list[int],
                       priorities: dict[int, list[float]]) -> int:
        """
        LOWEST priority in the cycle yields. Every robot computes the same
        answer from the same broadcast data -- no negotiation round-trip.
        """
        if not cycle:
            return 0
        return max(cycle, key=lambda r: tuple(priorities.get(r, [1e9])))


def make_priority(priority_class: int, deadline: float, eta: float,
                  battery_soc: float, robot_id: int) -> list[float]:
    slack = (deadline - eta) if deadline > 0 else 1e6
    return priority_tuple(priority_class, slack, battery_soc, robot_id)


# --------------------------------------------------------------------------
# 5. Predictive segment reservation  --  do not enter a jam you can foresee
# --------------------------------------------------------------------------

def segment_windows(path: list[tuple[int, int]], t_start: float, v_nom: float,
                    wmap, horizon: float = SEG_HORIZON
                    ) -> list[tuple[int, float, float, int, int]]:
    """
    Turn an announced path into (segment_id, t_enter, t_exit, dx, dy) claims.

    A robot already broadcasts its path cells in `Intent`. Every peer can
    therefore derive, with no new message type, WHEN that robot will be inside
    each narrow aisle segment and WHICH WAY it will be going. That is the
    whole input to predicting a head-on jam before either robot has committed
    to it -- and predicting it is the only affordable fix, because once both
    are inside a one-robot-wide corridor no velocity solution exists.

    Direction is taken over the whole traversal, not cell to cell: a path that
    jogs sideways within an aisle still has one net direction through it.
    """
    out: list[tuple[int, float, float, int, int]] = []
    if not path:
        return out
    step = CELL_SIZE / max(0.05, v_nom)
    cur_seg, t_in, first, last = -1, t_start, None, None
    t = t_start
    for (cx, cy) in path:
        if t - t_start > horizon:
            break
        seg = wmap.aisle_at(cx, cy)
        if seg != cur_seg:
            if cur_seg >= 0 and first is not None:
                out.append((cur_seg, t_in, t,
                            (last[0] > first[0]) - (last[0] < first[0]),
                            (last[1] > first[1]) - (last[1] < first[1])))
            cur_seg, t_in, first = seg, t, (cx, cy)
        last = (cx, cy)
        t += step
    if cur_seg >= 0 and first is not None and last is not None:
        out.append((cur_seg, t_in, t,
                    (last[0] > first[0]) - (last[0] < first[0]),
                    (last[1] > first[1]) - (last[1] < first[1])))
    return out


def predicted_head_on(mine: list[tuple[int, float, float, int, int]],
                      theirs: dict[int, list[tuple[int, float, float, int, int]]],
                      tau: float = 1.0) -> tuple[int, int, float] | None:
    """
    Earliest segment where a peer and I are announced to be inside together,
    travelling OPPOSITE ways.

    Returns (segment_id, peer_id, my_entry_time) or None. Same-direction
    overlap is fine -- that is a convoy, and convoys resolve themselves.
    """
    best = None
    for (seg, t0, t1, dx, dy) in mine:
        for pid, wins in theirs.items():
            for (pseg, pt0, pt1, pdx, pdy) in wins:
                if pseg != seg:
                    continue
                if pt1 + tau < t0 or t1 + tau < pt0:
                    continue                      # not there at the same time
                if dx * pdx + dy * pdy >= 0:
                    continue                      # same way, or turning
                if best is None or t0 < best[2]:
                    best = (seg, pid, t0)
    return best


# --------------------------------------------------------------------------
# 6. Negotiated push-back  --  "you are in my way, one of us must reverse"
# --------------------------------------------------------------------------

def resolve_pushback(my_prio: list[float], my_progress: float,
                     their_prio: list[float], their_progress: float,
                     my_id: int, their_id: int) -> bool:
    """
    True if I am the one who must reverse.

    Evaluated identically by both robots from the same broadcast numbers, so
    they cannot both reverse (which wastes two manoeuvres and re-blocks the
    corridor) and cannot both proceed (which is a collision the brake then has
    to catch). Deterministic and total: priority, then progress, then id.

    `my_prio` is a priority TUPLE where lower means more urgent, so a larger
    tuple yields. Progress is metres driven on the current leg; more progress
    wins. The id tie-break only ever runs when two robots have identical
    priority and are within PROGRESS_EPS, and it is stable.
    """
    if tuple(my_prio) != tuple(their_prio):
        return tuple(my_prio) > tuple(their_prio)
    if abs(my_progress - their_progress) > PROGRESS_EPS:
        return my_progress < their_progress
    return my_id > their_id
