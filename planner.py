"""
Space-time A* with congestion cost and peer reservations.

Architecture doc sections E.3 (congestion) and C.2 (planner choice).

Key correctness point: congestion goes into g() ONLY. The heuristic h()
stays pure distance/v_max so it remains admissible and A* stays correct.
Putting congestion into h is the classic way to silently break optimality.
"""

from __future__ import annotations

import heapq
import math

from amr_msgs import Reservation
from warehouse_map import WarehouseMap, CELL_SIZE

# cost weights (architecture doc E.3)
KAPPA = 2.0        # congestion weight
RHO = 0.9          # staleness decay per second
TAU = 1.5          # temporal buffer, seconds
LAMBDA_BLOCK = 50.0  # blocked-cell penalty

# CLEARANCE. The reservation table matched cells EXACTLY, so two paths could
# run two cells apart and the planner saw no cost at all -- but cells are
# 0.50 m and the robot's circumscribed diameter is 1.40 m, so two robots two
# cells apart on each axis are 1.41 m apart, i.e. exactly at the near-miss
# boundary. The planner was blind to the only separation that matters.
#
# Occupancy is therefore smeared over a neighbourhood: a peer's claimed cell
# still costs full, and cells near it cost a fraction that falls off with
# distance. It is a nudge, not a wall -- a hard exclusion would make the
# 3-cell-wide aisles unplannable, since two robots physically cannot be
# 1.40 m apart inside a 1.50 m corridor.
#
# Swept over 6 seeds x 180 s. Radius 3 is the best of the four, and the gain
# is in the aisles rather than in the headline count:
#   radius 0   91 near misses (53 narrow / 38 open)  2 collisions  56 tasks
#   radius 1  103              (42 / 61)             1             59
#   radius 2   96              (31 / 65)             2             61
#   radius 3   90              (30 / 60)             1             63
# Narrow-aisle near misses fall 53 -> 30 and throughput rises 56 -> 63, so the
# per-task rate goes 1.63 -> 1.43. The open-area count RISES simply because
# more work gets done; the headline total barely moves.
#
# Radius 3 is NOT used despite scoring best on that sweep. It broke
# test_congestion_changes_route: smearing 1.5 m in every direction penalises
# every alternative route equally, so the cost landscape flattens and
# congestion stops discriminating between paths at all. That test exists to
# catch exactly "the congestion term is not really wired in", and it was
# right. Radius 3 also costs 122 ms per replan against radius 2's 40 ms.
# Radius 2 keeps the aisle benefit (53 -> 31) with the term still working.
CLEARANCE_CELLS = 2          # 1.0 m of influence around a peer's claim
CLEARANCE_FALLOFF = 0.45     # weight of a cell one step off the claim


class ReservationTable:
    """Peer space-time claims, with automatic staleness decay.

    Stale peer info decays smoothly instead of being special-cased --
    that is how the system degrades gracefully when comms get patchy.
    """

    def __init__(self) -> None:
        self._res: dict[int, tuple[list[Reservation], float]] = {}
        self._blocked: dict[tuple[int, int], tuple[float, float]] = {}

    def update(self, robot_id: int, reservations: list[Reservation],
               stamp: float) -> None:
        self._res[robot_id] = (reservations, stamp)

    def drop(self, robot_id: int) -> None:
        self._res.pop(robot_id, None)

    def mark_blocked(self, cx: int, cy: int, now: float,
                     confidence: float = 1.0) -> None:
        self._blocked[(cx, cy)] = (confidence, now)

    def blocked_belief(self, cx: int, cy: int, now: float) -> float:
        """Confidence decays 0.05/s so temporary obstructions self-clear."""
        e = self._blocked.get((cx, cy))
        if not e:
            return 0.0
        conf, t = e
        return max(0.0, conf - 0.05 * (now - t))

    def occupancy(self, cx: int, cy: int, t: float, now: float,
                  exclude: int = 0, radius: int = 0) -> float:
        """
        Weighted count of peers reserving this cell near time t.

        `radius` > 0 also counts claims on NEARBY cells, at a weight that
        falls off with distance. That is what lets the planner keep robots
        apart instead of merely keeping them out of the same cell: a 0.50 m
        cell is far smaller than a 1.40 m robot, so cell-exact avoidance
        guarantees nothing about clearance.

        radius=0 preserves the original exact-cell semantics.
        """
        total = 0.0
        for rid, (res_list, stamp) in self._res.items():
            if rid == exclude:
                continue
            age = max(0.0, now - stamp)
            w = RHO ** age                      # staleness decay
            if w < 0.05:
                continue
            best = 0.0
            for r in res_list:
                d = abs(r.cx - cx) + abs(r.cy - cy)
                if d > radius:
                    continue
                if r.t_exit + TAU < t or t + TAU < r.t_enter:
                    continue
                near = w * (1.0 if d == 0 else CLEARANCE_FALLOFF ** d)
                if near > best:
                    best = near
                    if d == 0:
                        break                   # cannot do better than exact
            total += best
        return total


def heuristic(cx: int, cy: int, gx: int, gy: int, v_max: float) -> float:
    """Admissible: straight-line distance at max speed. No congestion here."""
    d = math.hypot(gx - cx, gy - cy) * CELL_SIZE
    return d / v_max


# Space-time search (CLAUDE.md flaw 2).
#
# The closed set used to be set[tuple[int, int]] -- purely spatial -- while
# the cost function was time-dependent, because `t_arrive` drives the
# occupancy lookup. That combination is not a missing feature, it is a
# correctness bug: the first expansion of a cell won permanently, at whatever
# time it happened to be reached, so the search could never represent "wait
# here 5 s and then go straight through". Its only way to express avoidance
# was to go AROUND, which is why congestion-aware routing on its own (B1)
# measured +0.18 s against B0 -- inside noise. It bought detours and nothing
# else.
#
# State is now (x, y, t_bucket) with an explicit WAIT action.
TIME_BUCKET = 1.0      # s of time discretisation
WAIT_TIEBREAK = 1e-3   # prefer moving over waiting when costs are equal
# Horizon is derived per call, not fixed. A fixed 40 s cap silently returned
# "unreachable" for any goal further than 40 s away -- which on this map
# includes the corner-to-corner route in test_path_valid_and_connected. The
# horizon must scale with the journey: three times the free-running time
# plus a minute leaves ample room to wait without admitting junk states.
HORIZON_SLACK = 3.0
HORIZON_PAD = 60.0


def plan_st(wmap: WarehouseMap, start: tuple[int, int], goal: tuple[int, int],
            table: ReservationTable, robot_id: int, t_start: float, now: float,
            v_nom: float = 0.6, v_max: float = 0.8,
            congestion_aware: bool = True,
            max_expansions: int = 60000
            ) -> list[tuple[int, int, float]]:
    """
    Space-time A*. Returns [(cx, cy, t_arrive), ...] which MAY repeat a cell
    when the plan is to wait there. [] if unreachable.

    `heuristic` stays distance/v_max with no congestion term. Congestion goes
    into g() only -- putting it into h() is the classic way to silently break
    admissibility, and with a WAIT action available an inadmissible h would
    start returning arbitrarily bad plans rather than merely suboptimal ones.
    """
    sx, sy = start
    gx, gy = goal
    if not wmap.is_free(sx, sy) or not wmap.is_free(gx, gy):
        return []
    if start == goal:
        return [(sx, sy, t_start)]

    step_time = CELL_SIZE / v_nom
    max_buckets = int((HORIZON_SLACK * heuristic(sx, sy, gx, gy, v_nom)
                       + HORIZON_PAD) / TIME_BUCKET)
    s0 = (sx, sy, 0)
    open_heap: list[tuple[float, int, tuple[int, int, int]]] = [(0.0, 0, s0)]
    counter = 0
    came: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    g_score = {s0: 0.0}
    # Cost and elapsed time are NOT the same quantity: a congestion penalty
    # adds to cost without any clock passing. Reconstructing arrival times
    # from g_score would report a robot arriving later the busier its route
    # looked, which is exactly backwards. Tracked separately.
    elapsed_at = {s0: 0.0}
    closed: set[tuple[int, int, int]] = set()
    expansions = 0

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expansions += 1
        if expansions > max_expansions:
            break

        cx, cy, _ = cur
        if (cx, cy) == (gx, gy):
            out = []
            node = cur
            while True:
                out.append((node[0], node[1], t_start + elapsed_at[node]))
                if node not in came:
                    break
                node = came[node]
            return out[::-1]

        g_cur = g_score[cur]
        e_cur = elapsed_at[cur]
        t_here = t_start + e_cur

        succ: list[tuple[tuple[int, int], float, float]] = []
        worst_occ = 0.0
        for nx, ny in wmap.neighbors(cx, cy):
            t_arrive = t_here + step_time
            cost = step_time
            if congestion_aware:
                occ = table.occupancy(nx, ny, t_arrive, now, exclude=robot_id,
                                      radius=CLEARANCE_CELLS)
                worst_occ = max(worst_occ, occ)
                cost += KAPPA * (occ ** 2)
                cost += LAMBDA_BLOCK * table.blocked_belief(nx, ny, now)
            succ.append(((nx, ny), cost, step_time))

        # WAIT is only generated where it could possibly help. Offering it
        # everywhere multiplies the branching factor by 5/4 across the whole
        # map to buy nothing on the ~90% of cells no peer has claimed, and
        # the replan latency budget is real.
        if congestion_aware and worst_occ > 0.05:
            succ.append(((cx, cy), TIME_BUCKET + WAIT_TIEBREAK, TIME_BUCKET))

        for (nc, cost, elapsed) in succ:
            g_new = g_cur + cost
            e_new = e_cur + elapsed
            bucket = int(e_new / TIME_BUCKET)
            if bucket > max_buckets:
                continue
            ns = (nc[0], nc[1], bucket)
            if ns in closed:
                continue
            if g_new < g_score.get(ns, float("inf")):
                g_score[ns] = g_new
                elapsed_at[ns] = e_new
                came[ns] = cur
                counter += 1
                f = g_new + heuristic(nc[0], nc[1], gx, gy, v_max)
                heapq.heappush(open_heap, (f, counter, ns))

    return []


def plan(wmap: WarehouseMap, start: tuple[int, int], goal: tuple[int, int],
         table: ReservationTable, robot_id: int, t_start: float, now: float,
         v_nom: float = 0.6, v_max: float = 0.8,
         congestion_aware: bool = True,
         max_expansions: int = 60000) -> list[tuple[int, int]]:
    """
    Cell path from start to goal, or [] if unreachable.

    Waits are COLLAPSED here so consecutive cells stay strictly adjacent --
    that is the long-standing contract every caller and the dashboard rely
    on. Use `plan_st` when you need the schedule as well; the arrival times
    are what tell the robot to hold rather than to detour.

    Set congestion_aware=False for the plain-A* baseline (B0/B1).
    """
    st = plan_st(wmap, start, goal, table, robot_id, t_start, now,
                 v_nom, v_max, congestion_aware, max_expansions)
    out: list[tuple[int, int]] = []
    for (cx, cy, _t) in st:
        if not out or out[-1] != (cx, cy):
            out.append((cx, cy))
    return out


def schedule_of(st: list[tuple[int, int, float]]) -> list[float]:
    """
    Earliest DEPARTURE time for each cell of the collapsed path.

    A wait shows up in `plan_st` as the same cell repeated; the last of those
    repeats carries the time the robot may leave. Aligning one departure time
    per collapsed cell is what lets the speed layer execute a planned wait as
    a slowdown rather than as a stop.
    """
    out: list[float] = []
    last: tuple[int, int] | None = None
    for (cx, cy, t) in st:
        if last == (cx, cy):
            out[-1] = t
        else:
            out.append(t)
            last = (cx, cy)
    return out


def path_to_reservations(path: list[tuple[int, int]], t_start: float,
                         v_nom: float, horizon: float = 10.0
                         ) -> list[Reservation]:
    """
    Convert a path into space-time claims for the next `horizon` seconds.

    This is what gets broadcast in Intent. Publishing only the near horizon
    keeps messages small and avoids over-committing to a distant future that
    will be replanned anyway.
    """
    out: list[Reservation] = []
    step_time = CELL_SIZE / max(0.05, v_nom)
    t = t_start
    for (cx, cy) in path:
        if t - t_start > horizon:
            break
        out.append(Reservation(cx=cx, cy=cy, t_enter=t, t_exit=t + step_time))
        t += step_time
    return out


def path_length_m(path: list[tuple[int, int]]) -> float:
    return max(0, len(path) - 1) * CELL_SIZE


def congestion_cost(path: list[tuple[int, int]], table: ReservationTable,
                    robot_id: int, t_start: float, now: float,
                    v_nom: float) -> float:
    """C_cong for the bid function (doc E.2/E.3)."""
    step_time = CELL_SIZE / max(0.05, v_nom)
    total = 0.0
    t = t_start
    for (cx, cy) in path:
        occ = table.occupancy(cx, cy, t, now, exclude=robot_id)
        total += KAPPA * (occ ** 2)
        t += step_time
    return total
