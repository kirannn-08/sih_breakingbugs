"""
Synthetic dataset generator for the deadlock-risk model.

Run:  python3 synth_data.py            (writes data/deadlock_dataset.npz + .csv)

WHY SYNTHETIC, AND WHY FROM THIS SIMULATOR
------------------------------------------
There is no logged fleet to learn from, so the data has to be generated. The
question is what generates it, and the answer here is deliberately NOT the
generic 40x40 grid that rl/env.py uses. That grid shares no geometry with the
warehouse the system actually has to run in: different aisle widths, no
passing bays, no blind corners at the aisle mouths, no charging bays, no task
distribution funnelling traffic through one cross-aisle. A model trained on it
would be learning a different building's traffic.

So every sample here comes from `sim2d.Simulation` on `WarehouseMap` -- the
same map, the same footprint, the same safety supervisor, the same occluded
LiDAR that the deployed model will see. Domain randomisation is over things
that genuinely vary at run time (fleet size, task rate, packet loss, seed),
not over the building.

LABELLING: SELF-SUPERVISED FROM OUTCOMES, NOT FROM A HEURISTIC
--------------------------------------------------------------
A sample is positive if, within LABEL_HORIZON seconds, that robot actually
became wedged -- either it entered a wait-for cycle, or it stalled for longer
than T_STALL while still having somewhere to go. That is an OUTCOME, recorded
by replaying the run forwards, not a hand-written rule about what a deadlock
looks like. Labelling with a rule would teach the model to imitate the rule,
and the rule is already available for free.

The "still having somewhere to go" clause matters: a robot parked on a
charging bay is stationary for minutes and is not deadlocked. Without that
clause roughly a third of the positives are idle robots and the model learns
to predict parking.
"""

from __future__ import annotations

import csv
import os
import random

import numpy as np

from coordination import T_STALL
from features import FEATURE_NAMES, LABEL_HORIZON, deadlock_features
from sim2d import DT, Simulation

OUT_DIR = "data"
SAMPLE_EVERY = 0.5          # s between samples; DT resolution is redundant


def rollout(seed: int, n_robots: int, loss: float, task_interval: float,
            duration: float = 180.0
            ) -> tuple[list[np.ndarray], list[int], list[dict]]:
    """One randomised run. Returns (features, labels, meta-rows)."""
    sim = Simulation(n_robots=n_robots, seed=seed, loss_rate=loss)
    obs: list[np.ndarray] = []
    meta: list[dict] = []
    # per sample: (robot_id, t). wedged[(rid, t)] is filled in on the way past.
    stamps: list[tuple[int, float]] = []
    wedged: dict[tuple[int, float], bool] = {}
    # times at which each robot was observably wedged
    wedge_times: dict[int, list[float]] = {r.id: [] for r in sim.robots}

    next_task, next_sample = 1.0, 0.0
    while sim.t < duration:
        if sim.t >= next_task and sim.next_task_id <= 20:
            sim.announce_task()
            next_task = sim.t + task_interval
        sim.step()

        for r in sim.robots:
            # ground truth: in a cycle, or stalled with work still to do
            in_cycle = r.id in r.deadlock.blocked_set(r.deadlock.find_cycle())
            stalled = (r.stall_since is not None
                       and sim.t - r.stall_since > T_STALL
                       and r.path_idx < len(r.path))
            if in_cycle or stalled:
                wedge_times[r.id].append(sim.t)

        if sim.t >= next_sample:
            next_sample = sim.t + SAMPLE_EVERY
            for r in sim.robots:
                if not r.path or r.path_idx >= len(r.path):
                    continue            # no plan: nothing to be blocked on
                obs.append(deadlock_features(r, sim.t))
                stamps.append((r.id, sim.t))
                meta.append({"seed": seed, "n_robots": n_robots,
                             "loss": loss, "t": round(sim.t, 1),
                             "robot": r.id,
                             "cell_x": r.cell[0], "cell_y": r.cell[1],
                             "narrow": int(r.wmap.is_narrow(*r.cell)),
                             "blind": int(r.wmap.is_blind_corner(*r.cell))})

    labels = []
    for (rid, t) in stamps:
        hits = wedge_times[rid]
        labels.append(int(any(t < w <= t + LABEL_HORIZON for w in hits)))
    for m, y in zip(meta, labels):
        m["label"] = y
    return obs, labels, meta


def generate(n_runs: int = 60, seed0: int = 1000
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = random.Random(seed0)
    X: list[np.ndarray] = []
    Y: list[int] = []
    G: list[int] = []
    M: list[dict] = []
    for k in range(n_runs):
        n = rng.choice([2, 3, 4, 5, 6])
        loss = rng.choice([0.0, 0.0, 0.05, 0.10, 0.20])
        ti = rng.choice([4.0, 6.0, 8.0, 12.0])
        x, y, m = rollout(seed0 + k, n, loss, ti)
        X.extend(x)
        Y.extend(y)
        G.extend([k] * len(x))
        M.extend(m)
        print(f"  run {k+1:>2}/{n_runs}  n={n} loss={loss:.2f} interval={ti:>4} "
              f"-> {len(x):>5} samples, {sum(y):>4} positive")
    return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64),
            np.asarray(G, dtype=np.int64), M)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"generating synthetic deadlock data on WarehouseMap "
          f"(horizon {LABEL_HORIZON:.0f} s)...")
    X, Y, G, M = generate()
    npz = os.path.join(OUT_DIR, "deadlock_dataset.npz")
    # G is the run index. Samples 0.5 s apart inside one run are strongly
    # correlated, so a random train/test split leaks the test set into
    # training and inflates held-out accuracy. The split must be by RUN.
    np.savez_compressed(npz, X=X, Y=Y, G=G, names=np.array(FEATURE_NAMES))
    print(f"\n{len(X)} samples, {Y.mean():.1%} positive, {X.shape[1]} features")
    print(f"saved {npz}")

    csv_path = os.path.join(OUT_DIR, "deadlock_samples.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(M[0].keys()))
        w.writeheader()
        w.writerows(M)
    print(f"saved {csv_path}  ({len(M)} rows, provenance for every sample)")


if __name__ == "__main__":
    main()
