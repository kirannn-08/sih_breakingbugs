"""
Lifelong MAPF environment for the N>32 scaling study.

WHY A SEPARATE ENVIRONMENT
--------------------------
sim2d.py is a continuous-space, 3-5 robot simulator with real kinematics and
a safety supervisor. It is the right tool for the SIH deliverable and the
wrong tool for RL: it caps at 5 robots, runs ~1 s of wall clock per 180 s of
sim, and its terminal-deadlock bug would be the dominant training signal.

This is a discrete grid environment in the standard MAPF formulation --
vectorised over agents, ~10^4 steps/s -- so a 64-agent curriculum is
tractable. It deliberately shares NO code with sim2d.py. Nothing here may be
imported by the production path.

Formulation: lifelong (online) MAPF. An agent that reaches its goal is
immediately assigned a new one, so throughput is a steady-state rate rather
than a makespan. That matches a warehouse and, unlike one-shot MAPF, it does
not reward parking.
"""

from __future__ import annotations

import numpy as np

# 0=stay, 1=up, 2=down, 3=left, 4=right
ACTIONS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int32)
N_ACTIONS = len(ACTIONS)


def sih_warehouse_grid(tile: int = 1) -> np.ndarray:
    """
    THE ACTUAL MAP the fleet runs on, as a MAPF grid. 1 = blocked.

    The generic `warehouse_grid` below shares no geometry with the warehouse
    this system has to work in: different aisle widths, no passing-bay
    notches, no blind corners at the aisle mouths, no charging bays, and a
    task distribution that does not funnel traffic through one cross-aisle.
    A scaling study run on it measures a different building, so its numbers
    cannot be carried over to this one -- and the policy trained on it was
    learning another warehouse's traffic.

    `tile` replicates the map in both axes so the >32-agent end of the
    curriculum has somewhere to put the agents. The real map has 804 free
    cells, and 64 agents on 804 cells is a density no warehouse operates at.
    Tiling keeps the LOCAL geometry -- aisle width, mouth spacing, rack
    pitch -- which is what the policy actually observes, while giving the
    fleet room.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from warehouse_map import FREE, WarehouseMap

    m = WarehouseMap()
    base = np.array([[0 if m.grid[y][x] == FREE else 1 for x in range(m.w)]
                     for y in range(m.h)], dtype=np.int8)
    if tile > 1:
        base = np.tile(base, (tile, tile))
    # The perimeter must stay drivable or the tiles are disconnected islands.
    base[0, :] = base[-1, :] = base[:, 0] = base[:, -1] = 0
    return base


def warehouse_grid(w: int = 40, h: int = 40, rack_w: int = 2, rack_h: int = 6,
                   aisle: int = 2) -> np.ndarray:
    """Regular rack layout. 1 = blocked. Border is always free (perimeter loop)."""
    g = np.zeros((h, w), dtype=np.int8)
    for y in range(aisle + 1, h - rack_h - aisle, rack_h + aisle):
        for x in range(aisle + 1, w - rack_w - aisle, rack_w + aisle):
            g[y:y + rack_h, x:x + rack_w] = 1
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 0
    return g


class LifelongMAPF:
    """
    Vectorised over agents. Positions are (N,2) int arrays in (x,y).

    Collision model (standard MAPF, and stricter than sim2d):
      * vertex conflict -- two agents may never occupy one cell
      * edge  conflict -- two agents may never swap cells
    A move that would cause either is REJECTED and the agent stays put. So
    collisions cannot occur; congestion shows up as lost throughput instead.
    That is the honest way to compare policies -- a policy cannot win by
    trading safety for speed.
    """

    def __init__(self, grid: np.ndarray, n_agents: int, seed: int = 0,
                 obs_radius: int = 4):
        self.grid = grid
        self.h, self.w = grid.shape
        self.n = n_agents
        self.obs_radius = obs_radius
        self.rng = np.random.default_rng(seed)
        self.free = np.argwhere(grid == 0)[:, ::-1]      # (x,y)
        if len(self.free) < 2 * n_agents:
            raise ValueError(f"map too small for {n_agents} agents")
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> np.ndarray:
        idx = self.rng.choice(len(self.free), size=self.n, replace=False)
        self.pos = self.free[idx].copy()
        self.goal = self._sample_goals(self.pos)
        self.t = 0
        self.delivered = 0
        self.blocked_moves = 0
        self.steps_since_progress = np.zeros(self.n, dtype=np.int32)
        return self.observe()

    def _sample_goals(self, avoid: np.ndarray) -> np.ndarray:
        idx = self.rng.choice(len(self.free), size=self.n, replace=True)
        g = self.free[idx].copy()
        same = np.all(g == avoid, axis=1)
        while same.any():
            g[same] = self.free[self.rng.choice(len(self.free), size=same.sum())]
            same = np.all(g == avoid, axis=1)
        return g

    # -- dynamics ----------------------------------------------------------

    def step(self, actions: np.ndarray):
        """actions: (N,) ints. Returns (obs, reward, info)."""
        prev = self.pos.copy()
        prop = self.pos + ACTIONS[actions]

        # static obstacles + bounds
        oob = ((prop[:, 0] < 0) | (prop[:, 0] >= self.w) |
               (prop[:, 1] < 0) | (prop[:, 1] >= self.h))
        prop[oob] = prev[oob]
        blocked = self.grid[prop[:, 1], prop[:, 0]] == 1
        prop[blocked] = prev[blocked]

        # vertex + edge conflicts, resolved by iterated rejection so that a
        # rejected agent cannot itself displace someone who had right of way
        for _ in range(4):
            keys = prop[:, 1] * self.w + prop[:, 0]
            order = np.argsort(keys, kind="stable")
            dup = np.zeros(self.n, dtype=bool)
            s = keys[order]
            same = np.flatnonzero(s[1:] == s[:-1])
            dup[order[same]] = True
            dup[order[same + 1]] = True

            pk, ck = prev[:, 1] * self.w + prev[:, 0], keys
            swap = np.zeros(self.n, dtype=bool)
            lookup = {int(k): i for i, k in enumerate(pk)}
            for i in range(self.n):
                j = lookup.get(int(ck[i]), -1)
                if j >= 0 and j != i and ck[j] == pk[i]:
                    swap[i] = swap[j] = True

            bad = dup | swap
            moved = np.any(prop != prev, axis=1)
            revert = bad & moved
            if not revert.any():
                break
            prop[revert] = prev[revert]

        self.blocked_moves += int(np.sum(np.all(prop == prev, axis=1) &
                                         (actions != 0)))
        self.pos = prop

        at_goal = np.all(self.pos == self.goal, axis=1)
        n_done = int(at_goal.sum())
        self.delivered += n_done
        if n_done:
            new = self._sample_goals(self.pos)
            self.goal[at_goal] = new[at_goal]

        prev_d = np.abs(prev - self.goal).sum(1)
        new_d = np.abs(self.pos - self.goal).sum(1)
        self.steps_since_progress = np.where(new_d < prev_d, 0,
                                             self.steps_since_progress + 1)

        # reward: progress shaping + delivery bonus - time - stall penalty
        rew = (prev_d - new_d).astype(np.float32) * 0.3
        rew += at_goal.astype(np.float32) * 5.0
        rew -= 0.05
        rew -= (self.steps_since_progress > 12).astype(np.float32) * 0.25

        self.t += 1
        info = {"delivered": self.delivered, "n_done": n_done,
                "blocked": self.blocked_moves,
                "stalled": int((self.steps_since_progress > 12).sum())}
        return self.observe(), rew, info

    # -- observation -------------------------------------------------------

    def observe(self) -> np.ndarray:
        """
        (N, 4, K, K) local egocentric stack + (N, 6) scalar features.
        Returned flattened as (N, 4*K*K + 6) so the numpy MLP can eat it.

        Channels: obstacles | peers | peer goals | own goal projection.
        Partial observability is the point -- this is what a real AMR can
        know from LiDAR plus the broadcast peer state in COMMS_INTERFACE.md.
        """
        r = self.obs_radius
        k = 2 * r + 1
        n = self.n
        obs = np.zeros((n, 4, k, k), dtype=np.float32)

        pad = np.ones((self.h + 2 * r, self.w + 2 * r), dtype=np.float32)
        pad[r:r + self.h, r:r + self.w] = self.grid
        occ = np.zeros_like(pad)
        occ[self.pos[:, 1] + r, self.pos[:, 0] + r] = 1.0
        gl = np.zeros_like(pad)
        gl[self.goal[:, 1] + r, self.goal[:, 0] + r] = 1.0

        for i in range(n):
            x, y = self.pos[i]
            obs[i, 0] = pad[y:y + k, x:x + k]
            obs[i, 1] = occ[y:y + k, x:x + k]
            obs[i, 2] = gl[y:y + k, x:x + k]
            obs[i, 1, r, r] = 0.0                      # don't see yourself

        d = (self.goal - self.pos).astype(np.float32)
        norm = np.maximum(1.0, np.abs(d).sum(1, keepdims=True))
        gx, gy = d[:, 0] / norm[:, 0], d[:, 1] / norm[:, 0]
        for i in range(n):
            obs[i, 3, r, r] = 1.0
        scal = np.stack([
            gx, gy,
            np.clip(np.abs(d).sum(1) / (self.w + self.h), 0, 1),
            np.clip(self.steps_since_progress / 20.0, 0, 1),
            np.full(n, self.n / 64.0, dtype=np.float32),
            np.clip(obs[:, 1].reshape(n, -1).sum(1) / 8.0, 0, 1),   # local density
        ], axis=1).astype(np.float32)
        return np.concatenate([obs.reshape(n, -1), scal], axis=1)

    @property
    def obs_dim(self) -> int:
        k = 2 * self.obs_radius + 1
        return 4 * k * k + 6

    def throughput(self) -> float:
        """Deliveries per agent per 100 steps -- the headline metric."""
        return 100.0 * self.delivered / max(1, self.t) / self.n
