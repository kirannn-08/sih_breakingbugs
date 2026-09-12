"""
Local observation for the deadlock-risk model.

ONE EXTRACTOR, TWO CALLERS
--------------------------
`synth_data.py` calls this to build the training set and `sim2d.py` calls the
identical function at run time. Writing the features twice -- once for the
dataset and once for inference -- is the standard way to ship a model that
scores 90% offline and does nothing useful online, and this repo has already
shipped enough symbols that looked like working code.

WHY THESE FEATURES AND NOT A GRID
---------------------------------
`learned.py` uses a 7x7x2 occupancy window, which suits an action policy that
has to steer. This model answers a different question -- "am I about to get
wedged?" -- and the evidence for that is relational, not pictorial: who is
converging on the corridor I am about to enter, from which side, how fast,
and how long have they been there. A raw occupancy grid makes a small MLP
learn those relations from scratch; handing them over directly is what keeps
this at a few thousand parameters and inference under a millisecond.

STRICTLY LOCAL
--------------
Everything here comes from the robot's own state, its own LiDAR tracks, or
peer broadcasts it has actually received. Nothing reads another robot's
variables. If a peer has gone quiet its slot is zero-filled and the model
sees a fleet it cannot hear -- which is the honest input, and the case the
model most needs to handle, since a robot that cannot hear its peers is
exactly the one most likely to drive into a jam.
"""

from __future__ import annotations

import math

import numpy as np

K_PEERS = 3                 # nearest peers described individually
WIN = 5                     # static occupancy window (odd)

# index layout, kept explicit so the CSV dump and any later debugging agree
FEATURE_NAMES: list[str] = (
    [f"occ_{i}" for i in range(WIN * WIN)]
    + ["goal_dx", "goal_dy", "goal_dist",
       "in_narrow", "at_blind_corner", "in_segment",
       "my_speed", "stall_age", "region_age",
       "peers_in_my_segment", "occ_ahead", "n_peers_heard", "comms_degraded",
       "lidar_gap", "lidar_speed", "lidar_static_age"]
    + [f"p{k}_{n}" for k in range(K_PEERS)
       for n in ("dx", "dy", "dvx", "dvy", "opposed", "same_seg", "heard")]
)
FEAT_DIM = len(FEATURE_NAMES)

# horizon the label looks ahead over. 10 s is ~8 m of travel: long enough
# that avoiding the jam is still possible, short enough that the outcome is
# attributable to the state rather than to everything that happened after.
LABEL_HORIZON = 10.0


def _norm(v: float, scale: float) -> float:
    """Squash to roughly [-1, 1] without clipping information to a hard edge."""
    return math.tanh(v / scale)


def deadlock_features(robot, now: float) -> np.ndarray:
    """Build the observation vector for one robot at one instant."""
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    wmap = robot.wmap
    cx, cy = robot.cell
    half = WIN // 2

    i = 0
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            f[i] = 0.0 if wmap.is_free(cx + dx, cy + dy) else 1.0
            i += 1

    if robot.goal:
        gx, gy = robot.goal
        d = math.hypot(gx - cx, gy - cy)
        f[i] = (gx - cx) / max(1.0, d)
        f[i + 1] = (gy - cy) / max(1.0, d)
        f[i + 2] = _norm(d, 20.0)
    i += 3

    f[i] = 1.0 if wmap.is_narrow(cx, cy) else 0.0
    f[i + 1] = 1.0 if wmap.is_blind_corner(cx, cy) else 0.0
    f[i + 2] = 1.0 if robot.region >= 0 else 0.0
    i += 3

    f[i] = robot.v / 0.8
    f[i + 1] = _norm(0.0 if robot.stall_since is None
                     else now - robot.stall_since, 5.0)
    f[i + 2] = _norm(0.0 if robot.region < 0
                     else now - robot.region_entered_at, 10.0)
    i += 3

    fresh = [(pid, st) for pid, st in robot.peers.items()
             if now - robot.peer_seen_at.get(pid, -1e9) <= 2.0]
    same_seg = 0
    for pid, st in fresh:
        if wmap.aisle_at(*wmap.to_cell(st.x, st.y)) == robot.region >= 0:
            same_seg += 1
    f[i] = same_seg / 4.0
    # how contested is the cell I am about to enter, by peers' own claims
    occ_ahead = 0.0
    if robot.path_idx < len(robot.path):
        nx, ny = robot.path[robot.path_idx]
        occ_ahead = robot.table.occupancy(nx, ny, now, now, exclude=robot.id)
    f[i + 1] = _norm(occ_ahead, 2.0)
    f[i + 2] = len(fresh) / 4.0
    f[i + 3] = 1.0 if robot.degraded else 0.0
    i += 4

    lane = robot.lidar.lane_tracks(robot.x, robot.y, robot.theta,
                                   0.98, 3.0, now)
    if lane:
        fwd, tr = lane[0]
        f[i] = _norm(fwd, 3.0)
        f[i + 1] = tr.speed() / 0.8
        f[i + 2] = _norm(tr.static_for(now), 5.0)
    else:
        f[i] = 1.0            # nothing in my lane reads as "maximally clear"
    i += 3

    hx, hy = math.cos(robot.theta), math.sin(robot.theta)
    fresh.sort(key=lambda p: math.hypot(p[1].x - robot.x, p[1].y - robot.y))
    for k in range(K_PEERS):
        base = i + k * 7
        if k >= len(fresh):
            continue                      # zero-filled: a peer not heard
        pid, st = fresh[k]
        dx, dy = st.x - robot.x, st.y - robot.y
        f[base] = _norm(dx, 4.0)
        f[base + 1] = _norm(dy, 4.0)
        f[base + 2] = _norm(st.vx - robot.v * hx, 0.8)
        f[base + 3] = _norm(st.vy - robot.v * hy, 0.8)
        sp = math.hypot(st.vx, st.vy)
        # Opposed headings in a shared corridor are the single strongest
        # precursor of a wedge on this map, so it is handed over directly
        # rather than left to be inferred from four velocity components.
        f[base + 4] = (-(st.vx * hx + st.vy * hy) / sp) if sp > 0.05 else 0.0
        pseg = wmap.aisle_at(*wmap.to_cell(st.x, st.y))
        f[base + 5] = 1.0 if (pseg >= 0 and pseg == robot.region) else 0.0
        f[base + 6] = 1.0
    return f
