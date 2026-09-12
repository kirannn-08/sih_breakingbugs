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
