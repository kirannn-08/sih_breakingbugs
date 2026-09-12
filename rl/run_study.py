"""
Scaling study: does a decentralized learned policy beat classical planning,
and at what fleet size? Run: python3 run_study.py
"""
from __future__ import annotations
import json, time
import numpy as np
from env import LifelongMAPF, sih_warehouse_grid
from baselines import greedy_jitter, WHCAStar
from policy import MLP

STEPS, SEEDS = 400, 3

def run(grid, n, make_policy, seed):
    e = LifelongMAPF(grid, n, seed=seed)
    pol = make_policy(e)
    rng = np.random.default_rng(seed)
    obs = e.reset()
    for _ in range(STEPS):
        a = np.asarray(pol(e, rng, obs) if pol.__code__.co_argcount == 3
                       else pol(e, rng))
        obs, _, _ = e.step(a)
    return e.throughput(), e.blocked_moves

def main():
    grid = sih_warehouse_grid(tile=1)
    free = int((grid == 0).sum())
    net = MLP.load("policy_marl.npz")

    def learned(e):
        def f(env, rng, obs=None):
            if obs is None:
                obs = env.observe()
            return net.act(obs)
        f.__code__ = f.__code__          # keep 3-arg signature
        return f

    rows = []
    print(f"grid {grid.shape[1]}x{grid.shape[0]} (this warehouse), "
          f"{free} free cells, {STEPS} steps, {SEEDS} seeds\n")
    print(f"{'N':>4}{'density':>9}{'greedy+j':>11}{'WHCA*':>9}{'learned':>9}"
          f"{'learned/WHCA*':>15}{'blocked(L)':>12}")
    for n in (8, 16, 32, 48, 64, 96, 128):
        res = {}
        for name, mk in (("greedy", lambda e: greedy_jitter),
                         ("whca", lambda e: WHCAStar(e)),
                         ("learned", learned)):
            tp, bl = [], []
            for s in range(1, SEEDS + 1):
                e = LifelongMAPF(grid, n, seed=s)
                pol = mk(e); rng = np.random.default_rng(s)
                obs = e.reset()
                for _ in range(STEPS):
                    a = net.act(obs) if name == "learned" else np.asarray(pol(e, rng))
                    obs, _, _ = e.step(np.asarray(a))
                tp.append(e.throughput()); bl.append(e.blocked_moves)
            res[name] = (float(np.mean(tp)), float(np.mean(bl)))
        ratio = res["learned"][0] / max(1e-9, res["whca"][0])
        rows.append(dict(n=n, density=n / free,
                         greedy=res["greedy"][0], whca=res["whca"][0],
                         learned=res["learned"][0], ratio=ratio,
                         blocked_learned=res["learned"][1]))
        print(f"{n:>4}{100*n/free:>8.1f}%{res['greedy'][0]:>11.3f}"
              f"{res['whca'][0]:>9.3f}{res['learned'][0]:>9.3f}"
              f"{ratio:>14.0%}{res['learned'][1]:>12.0f}")

    e = LifelongMAPF(grid, 32, seed=1); obs = e.reset()
    t0 = time.time()
    for _ in range(200): net.act(obs)
    dt = (time.time() - t0) / 200 / 32
    print(f"\ninference: {dt*1e6:.1f} us per agent per step  "
          f"({net.n_params:,} params, numpy, single core)")
    json.dump(rows, open("results.json", "w"), indent=2)
    import csv, os
    os.makedirs("../results", exist_ok=True)
    with open("../results/mapf_scaling.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote results.json and ../results/mapf_scaling.csv")

if __name__ == "__main__":
    main()
