"""
LiDAR track layer -- decisions from local sensing, not just from peer radio.

WHY THIS EXISTS
---------------
Before this module the only consumer of LiDAR was `safety_supervisor` (an
instantaneous geometric brake) and `orca_adjust`. Every *decision* -- who
yields, where to reroute, whether a corridor is safe to enter -- was made
purely from `Intent` and `RobotState` broadcasts. That has two consequences
this project cannot afford:

  1. A robot whose radio is down makes no coordination decisions at all. It
     brakes and it stalls. The "graceful degradation" claim was carried
     entirely by the brake.
  2. A stopped robot that never *sent* an Obstacle message (crashed, out of
     battery, a dropped pallet, a human) is invisible to the routing layer
     even while it sits in plain view of the LiDAR.

So this keeps TRACKS: short-lived, id-less estimates of what the sensor has
actually seen, with a velocity derived by finite difference. Tracks feed the
routing and deadlock layers directly. Nothing here reads peer state.

ID-LESS ON PURPOSE
------------------
A real LiDAR returns points, not robot ids. Associating a track with a
`robot_id` would smuggle comms knowledge into the sensor path and quietly
recreate the omniscience this module is meant to remove. Association here is
nearest-neighbour in space, and a track that drops out is forgotten.

DETERMINISM
-----------
No noise is injected. `test_deterministic_given_seed` is what makes the
benchmark reproducible, and a stochastic sensor would break it for no
scientific gain -- sensor noise belongs in the ROS 2 / Gazebo transfer,
where a real driver supplies it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# association gate: a return further than this from a track's prediction
# starts a new track instead of updating the old one.
GATE_M = 0.9

V_STILL = 0.08          # m/s below which a track counts as stationary
T_DROP = 1.0            # s without a return before a track is forgotten
T_CONFIRM = 0.3         # s of continuous returns before a track is trusted
T_STATIC = 2.5          # s stationary before a track is an OBSTACLE, not traffic
ALPHA_V = 0.35          # EMA on velocity; raw finite differences are too noisy

# Confidence a LiDAR-derived blockage is written into the reservation table
# with. Deliberately LOWER than a broadcast Obstacle (1.0): the sensor knows
# something is there, not that it intends to stay. It decays at 0.05/s like
# any other belief, so a peer that moves on clears itself.
LIDAR_BLOCK_CONF = 0.6


@dataclass
class Track:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    still_since: float | None = None
    hits: int = 0

    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def confirmed(self, now: float) -> bool:
        return self.hits >= 2 and (now - self.first_seen) >= T_CONFIRM

    def static_for(self, now: float) -> float:
        """Seconds this track has been stationary. 0.0 if it is moving."""
        if self.still_since is None:
            return 0.0
        return now - self.still_since

    def predict(self, dt: float) -> tuple[float, float]:
        """Constant-velocity extrapolation. Honest for ~1-2 s, no further."""
        return self.x + self.vx * dt, self.y + self.vy * dt


class LidarTracker:
    """Per-robot. Consumes (x, y) returns in the WORLD frame."""

    def __init__(self) -> None:
        self.tracks: list[Track] = []
        # instrumentation. A guard that never fires is worse than no guard,
        # so every rule this module adds is counted and asserted on.
        self.n_returns = 0
        self.n_new_tracks = 0
        self.n_static_seen = 0

    def update(self, returns: list[tuple[float, float]], now: float,
               dt: float) -> None:
        used: set[int] = set()
        for (rx, ry) in returns:
            self.n_returns += 1
            best_i, best_d = -1, GATE_M
            for i, t in enumerate(self.tracks):
                if i in used:
                    continue
                px, py = t.predict(now - t.last_seen)
                d = math.hypot(rx - px, ry - py)
                if d < best_d:
                    best_i, best_d = i, d
            if best_i < 0:
                self.tracks.append(Track(x=rx, y=ry, first_seen=now,
                                         last_seen=now, hits=1,
                                         still_since=now))
                used.add(len(self.tracks) - 1)
                self.n_new_tracks += 1
                continue
            t = self.tracks[best_i]
            span = max(1e-3, now - t.last_seen)
            vx_raw, vy_raw = (rx - t.x) / span, (ry - t.y) / span
            t.vx += ALPHA_V * (vx_raw - t.vx)
            t.vy += ALPHA_V * (vy_raw - t.vy)
            t.x, t.y, t.last_seen = rx, ry, now
            t.hits += 1
            if t.speed() < V_STILL:
                if t.still_since is None:
                    t.still_since = now
            else:
                t.still_since = None
            used.add(best_i)

        self.tracks = [t for t in self.tracks if now - t.last_seen <= T_DROP]
        self.n_static_seen += sum(1 for t in self.tracks
                                  if t.static_for(now) > T_STATIC)

    # -- queries used by the decision layers -------------------------------

    def lane_tracks(self, x: float, y: float, theta: float,
                    half_w: float, max_fwd: float,
                    now: float) -> list[tuple[float, Track]]:
        """
        Confirmed tracks AHEAD of me and inside my lane, nearest first.

        Body-frame, same convention as the safety supervisor: `fwd` along
        heading, `lat` across it. A peer beside me is not in my way.
        """
        out = []
        hx, hy = math.cos(theta), math.sin(theta)
        for t in self.tracks:
            if not t.confirmed(now):
                continue
            dx, dy = t.x - x, t.y - y
            fwd = dx * hx + dy * hy
            lat = -dx * hy + dy * hx
            if fwd <= 0.0 or fwd > max_fwd or abs(lat) >= half_w:
                continue
            out.append((fwd, t))
        out.sort(key=lambda p: p[0])
        return out

    def blocking_tracks(self, x: float, y: float, theta: float,
                        half_w: float, max_fwd: float,
                        now: float) -> list[Track]:
        """In-lane tracks that have been STATIONARY long enough to be obstacles."""
        return [t for _, t in self.lane_tracks(x, y, theta, half_w, max_fwd, now)
                if t.static_for(now) > T_STATIC]

    def closing_track(self, x: float, y: float, theta: float,
                      half_w: float, max_fwd: float,
                      now: float) -> tuple[Track, float] | None:
        """
        Nearest in-lane track we are closing on, with time-to-contact.

        Approach speed is measured against MY heading only -- I do not know
        my own velocity vector from the sensor, so the closing rate here is
        the peer's component along my lane. That under-estimates closure when
        I am moving, which is the safe direction to be wrong in for a rule
        that can only ever slow me down.
        """
        for fwd, t in self.lane_tracks(x, y, theta, half_w, max_fwd, now):
            hx, hy = math.cos(theta), math.sin(theta)
            approach = -(t.vx * hx + t.vy * hy)      # +ve = coming at me
            if approach <= 0.05:
                continue
            return t, fwd / approach
        return None

    def occupied_cells(self, wmap, now: float,
                       only_static: bool = True) -> list[tuple[int, int]]:
        """Grid cells covered by tracks -- what gets written into the table."""
        out = []
        for t in self.tracks:
            if not t.confirmed(now):
                continue
            if only_static and t.static_for(now) <= T_STATIC:
                continue
            out.append(wmap.to_cell(t.x, t.y))
        return out

    def stats(self) -> dict:
        return {"returns": self.n_returns,
                "tracks_created": self.n_new_tracks,
                "static_track_ticks": self.n_static_seen,
                "live_tracks": len(self.tracks)}
