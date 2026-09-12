"""
Benchmark harness.  This produces the number you defend to judges.

Arms (architecture doc K.1) -- all share map, tasks, seeds, speeds:

  B0  stop-and-wait           <- the problem statement's baseline
  B1  + congestion-aware routing
  B2  + speed adaptation      <- the headline mechanism
  B3  full system
  B4  full system @ 20% packet loss   (resilience, not speed)
  B5  full system + LIFO deadlock stack OFF   (ablation)
  B6  full system + learned deadlock-risk model ON

Writes results/benchmark.csv -- ONE ROW PER ARM PER SEED, not just the
aggregate. The aggregate hides the thing that matters most on this map:
outcomes are bimodal, because a seed either jams or it does not, and a mean
over a bimodal distribution describes no run that actually happened.

TWO HEADLINE METRICS, AND THEY DISAGREE
---------------------------------------
`avg_task_time` is averaged over COMPLETED tasks only, so a system that
finishes the hard tasks its baseline gives up on is PENALISED for finishing
them -- the extra completions are the slow ones and they drag the mean up.
On seed 3 the full system completes 5 tasks against the baseline's 1.

`throughput` has no such bias. Both are reported on every row. Where they
disagree, throughput is the one to believe, and the disagreement itself is
worth showing rather than resolving by picking the flattering one.

Run:  python3 run_benchmark.py [n_seeds]
"""

from __future__ import annotations

import csv
import json
import os
import statistics as stats
import sys

from sim2d import Simulation

ARMS = {
    "B0_stop_and_wait":  dict(congestion=False, speed_adapt=False, loss_rate=0.0),
    "B1_congestion":     dict(congestion=True,  speed_adapt=False, loss_rate=0.0),
    "B2_speed_adapt":    dict(congestion=False, speed_adapt=True,  loss_rate=0.0),
    "B3_full":           dict(congestion=True,  speed_adapt=True,  loss_rate=0.0),
    "B4_full_20pct_loss": dict(congestion=True, speed_adapt=True,  loss_rate=0.20),
    "B5_no_lifo_stack": dict(congestion=True, speed_adapt=True, loss_rate=0.0,
                             lifo_stack=False),
    "B6_full_risk_model": dict(congestion=True, speed_adapt=True, loss_rate=0.0,
                               use_risk_model=True),
}

# Columns written per seed. Every rule added by this work is counted here,
# because a rule whose counter is always zero is dead code wearing a costume.
CSV_FIELDS = [
    "arm", "seed", "n_robots", "duration_s",
    "tasks_completed", "throughput_per_min", "avg_task_time",
    "collisions", "collision_ticks", "near_misses",
    "deadlocks", "avg_recovery_s", "backouts", "backouts_done",
    "entry_deferrals", "risk_deferrals", "planned_waits", "waits_executed",
    "blind_slowdowns", "lidar_occluded", "lidar_blocks", "lidar_replans",
    "lidar_decisions", "full_stops", "time_stopped", "distance_m", "replans",
    "comms_sent", "comms_dropped", "comms_loss_pct",
]


def run_arm(name: str, cfg: dict, seeds: list[int],
            n_robots: int = 4, duration: float = 180.0,
            per_seed: list[dict] | None = None) -> dict:
    rows = []
    for s in seeds:
        sim = Simulation(n_robots=n_robots, seed=s, **cfg)
        r = sim.run(duration=duration)
        rows.append(r)
        if per_seed is not None:
            c = r.get("comms", {})
            rec = {k: r.get(k, 0) for k in CSV_FIELDS}
            rec.update(arm=name, seed=s, n_robots=n_robots,
                       duration_s=duration,
                       comms_sent=c.get("sent", 0),
                       comms_dropped=c.get("dropped", 0),
                       comms_loss_pct=c.get("loss_pct", 0.0))
            per_seed.append(rec)

    def agg(key):
        vals = [r[key] for r in rows]
        return (round(stats.mean(vals), 2),
                round(stats.pstdev(vals), 2) if len(vals) > 1 else 0.0)

    tp_m, tp_s = agg("throughput_per_min")
    tt_m, tt_s = agg("avg_task_time")
    return {
        "arm": name,
        "throughput_per_min": tp_m, "throughput_std": tp_s,
        "avg_task_time": tt_m, "task_time_std": tt_s,
        "tasks_completed": agg("tasks_completed")[0],
        "collisions": sum(r["collisions"] for r in rows),
        "deadlocks": agg("deadlocks")[0],
        "full_stops": agg("full_stops")[0],
        "time_stopped": agg("time_stopped")[0],
        "seeds": len(seeds),
    }


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seeds = list(range(1, n_seeds + 1))
    print(f"Running {len(ARMS)} arms x {n_seeds} seeds, 4 robots\n")

    results = {}
    per_seed: list[dict] = []
    for name, cfg in ARMS.items():
        print(f"  {name} ...", flush=True)
        results[name] = run_arm(name, cfg, seeds, per_seed=per_seed)

    base = results["B0_stop_and_wait"]
    print("\n" + "=" * 86)
    print(f"{'arm':<22}{'thru/min':>10}{'tasks':>7}{'task_s':>9}"
          f"{'stopped_s':>11}{'coll':>6}{'dlk':>6}{'vs B0':>12}")
    print("-" * 86)
    for name, r in results.items():
        if r["avg_task_time"] > 0 and base["avg_task_time"] > 0:
            impr = 100.0 * (base["avg_task_time"] - r["avg_task_time"]) \
                / base["avg_task_time"]
        else:
            impr = 0.0
        tag = "baseline" if name == "B0_stop_and_wait" else f"{impr:+.1f}%"
        print(f"{name:<22}{r['throughput_per_min']:>10.2f}"
              f"{r['tasks_completed']:>7.1f}{r['avg_task_time']:>9.1f}"
              f"{r['time_stopped']:>11.1f}{r['collisions']:>6}"
              f"{r['deadlocks']:>6.1f}{tag:>12}")
    print("=" * 86)

    b0_tasks = base["tasks_completed"]
    b3_tasks = results["B3_full"]["tasks_completed"]
    if b0_tasks:
        print(f"THROUGHPUT (unbiased): {b0_tasks:.1f} -> {b3_tasks:.1f} tasks "
              f"per run = {100*(b3_tasks-b0_tasks)/b0_tasks:+.1f}%")

    full = results["B3_full"]
    if base["avg_task_time"] > 0:
        gain = 100.0 * (base["avg_task_time"] - full["avg_task_time"]) \
            / base["avg_task_time"]
        print(f"\nHEADLINE: task-completion time {gain:+.1f}% vs stop-and-wait")
        print(f"          collisions across ALL runs: {full['collisions']}")
        print(f"          target is >=20% reduction + zero collisions")
        print("\nPASS" if gain >= 20 and full["collisions"] == 0
              else "\nNOT YET AT TARGET")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(per_seed)
    print(f"\nwrote benchmark_results.json and results/benchmark.csv "
          f"({len(per_seed)} rows)")


if __name__ == "__main__":
    main()
