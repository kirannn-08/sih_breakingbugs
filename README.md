# Edge-AI Distributed Fleet Coordination for AMRs

**Smart India Hackathon — Problem Statement 26123**

A fully decentralized coordination stack for a fleet of warehouse Autonomous
Mobile Robots. No central path planner. Each robot decides its own task,
route, speed and conflict resolution onboard, and the fleet keeps operating
when the network fails.

---

## Results

4 robots · 10 seeds · identical map, task set and seeds across all arms

| Arm | Throughput /min | Avg task time (s) | Full stops | Collisions | vs B0 |
|---|---|---|---|---|---|
| B0 stop-and-wait *(PS baseline)* | 1.57 | 39.9 | 11.5 | 0 | — |
| B1 + congestion routing | 1.93 | 40.9 | 12.0 | 0 | -2.6% |
| B2 + speed adaptation | 1.97 | 33.1 | 4.2 | 0 | **+17.0%** |
| **B3 full system** | **1.67** | **31.1** | 3.8 | **0** | **+22.1%** |
| B4 full @ 20% packet loss | 1.90 | 33.4 | 3.0 | 0 | +16.2% |

### Success criteria

| Target | Result |
|---|---|
| Zero inter-robot collisions | ✅ **0** across all arms, all seeds |
| ≥20% task-time reduction vs stop-and-wait | ✅ **+22.1%** |
| Tests | ✅ **51/51 passing** |

**What the ablation shows:** speed adaptation is the dominant contributor
(+17.0% on its own). We can say precisely which mechanism earns the gain
rather than claiming the system works as a black box.

**Why B4 is now worse than B3, and why that is the honest number.** Task
allocation runs over the radio: each robot broadcasts its `Bid`, resolves
the auction from the bids it *actually received*, and the believed winner
broadcasts a `Claim`. Dropping 20% of packets therefore costs real
allocation time — median time from announcement to acceptance rises from
**0.40 s to 1.10 s** — and re-auctions cost throughput. An earlier version
of this table showed B4 *beating* B3, which was an artifact: allocation was
resolved centrally in-process and never touched the network, so packet loss
had nothing to degrade.

---

## The idea in one paragraph

Conventional fleets ask *"where is each robot?"* and route from a central
server. We broadcast **intent** — the space-time cells each robot will
occupy and when — so peers predict conflicts *before* they happen. When two
robots want the same aisle, the system computes a target arrival time and
derives the speed to achieve it, rather than stopping one robot. Stopping is
a computed last resort, not a reflex. That single mechanism is where the
throughput gain comes from.

---

## Architecture

```
+--------------------------------------------------+---------+
| L6 TASK       Auction / Claim / Release           | event   |
| L5 ROUTE      Space-time A* + congestion cost     | 0.5-2Hz |
|               learned ETA predictor  (Edge-AI)    |         |
| L4 COORDINATE Conflict prediction, priority       | 5 Hz    |
|               speed scheduling, deadlock cycles   |         |
| L3 AVOIDANCE  ORCA  (zone-gated)                  | 10-20Hz |
| L2 SAFETY     Deterministic. Geometric. FINAL.    | 30-50Hz |
| L1 CONTROL    Diff-drive velocity controller      | 20-50Hz |
| L0 PERCEPTION LiDAR / odom / IMU / localization   | 10-20Hz |
+--------------------------------------------------+---------+
```

**Safety invariant:** L2 is the only layer with authority over the motors,
and it can only ever *reduce* speed. No learned component sits below L4.
L2 runs on LiDAR, not comms — which is why the fleet stays collision-free
during a total network blackout.

---

## Repository layout

```
amr_fleet/
├── amr_msgs.py           FROZEN message contract  <- start here
├── warehouse_map.py      grid map, aisle segmentation, passing bays
├── comms.py              comms mediator: loss, latency, dead zones, link cuts
├── planner.py            space-time A*, congestion cost, reservations
├── coordination.py       auction, conflict prediction, speed adaptation, deadlock
├── learned.py            8.9k-param policy + hybrid fallback
├── sim2d.py              fast headless simulator + robot agent
├── train_policy.py       imitation training (~4 s end to end)
├── run_benchmark.py      5-arm benchmark harness
├── demo_scenarios.py     7 runnable demo scenarios
├── dashboard_server.py   live WebSocket & REST mission control server
├── web/                  web frontend (HTML5/Canvas/CSS/JS mission control)
├── tests/test_all.py     51 tests (+ test_dashboard.py)
├── docs/ENGINEERING_LOG.md   every bug we hit, with evidence
└── results/              captured test + benchmark output
```

---

## Quick start

```bash
pip install -r requirements.txt

python3 dashboard_server.py     # launch live web frontend (http://localhost:8080)
python3 warehouse_map.py        # render the map
python3 sim2d.py                # single run
python3 train_policy.py         # train the policy (~4 s)
python3 run_benchmark.py 10     # reproduce +22.1%
python3 demo_scenarios.py       # all 7 demo scenarios
python3 -m pytest tests/ -v     # 57 tests
```

Only runtime dependency is **numpy** (and **tornado** for the web dashboard).

---

## The learning component

A **8,901-parameter** policy maps a **7×7 local observation window** to a
movement action. It is trained by **imitation** from the full planner acting
as an expert with global information.

| Metric | Value |
|---|---|
| Training data | 9,565 self-generated state-action pairs |
| Training time | **0.8 s on CPU** |
| Held-out agreement | **91.8%** |
| Inference | **0.025 ms** (40 kHz capable) |
| Framework | pure numpy — no torch on a Raspberry Pi |

**Why imitation and not RL.** Reinforcement learning needs reward shaping,
exploration scheduling and stability work — 40–80 engineering hours with
failure modes that are hard to diagnose. More importantly, MARL's advantage
appears at high agent density; at 3–5 robots prioritized space-time planning
is already near-optimal, so a learned policy has little headroom to beat it.
Distilling from a planner that is already correct gives exact labels, plain
cross-entropy loss, and sub-second convergence. Same imitation component
PRIMAL uses (Sartoretti et al., RA-L 2019).

**The information gap is the point.** The expert sees the whole warehouse;
the student sees a 7×7 window. Reproducing 91.8% of its decisions from local
information alone *is* the decentralization result.

---

## Engineering log

`docs/ENGINEERING_LOG.md` documents all 7 real bugs hit during development,
with the measurements that exposed each one. Two worth reading:

- **A guard that never fires is worse than no guard.** Our corridor
  reservation rule was silently dead for two iterations because aisles were
  segmented per-column instead of across the corridor width.
- **Identical output after a behavioural change means the code path is
  dead.** Byte-identical benchmark results were the clue that found it.

---

## Scope

**Implemented:** decentralized auction, congestion-aware space-time planning,
P2P intent sharing, speed adaptation, corridor reservation, zone-gated ORCA,
deterministic safety supervisor, deadlock detection and recovery, dynamic
rerouting, degraded-mode operation, comms mediator, learned policy, benchmark
harness.

**Deliberately out of scope:** full MARL training (cost/benefit at N=5),
D\* Lite (incremental repair nullified by globally-changing congestion
costs), CBBA (bundle logic unused at 3–5 robots), SLAM (known map per the PS),
physical robots (Gazebo is the transfer path).

---

## Novelty

We claim **no novelty** in A\*, ORCA, auction protocols, imitation learning
or ROS 2 — these are established methods and we cite them. Our contribution
is the **system-level integration**: a decentralized stack where allocation,
routing, conflict prediction, speed negotiation, deadlock recovery and
degraded operation all run onboard, with a **measured** performance envelope
under communication loss.

---

## References

Sharon et al. 2015 (CBS) · Silver 2005 (WHCA\*) · Phillips & Likhachev 2011
(SIPP) · van den Berg et al. 2011 (ORCA) · Fox et al. 1997 (DWA) · Smith 1980
(Contract Net) · Gerkey & Matarić 2004 (MRTA taxonomy) · Sartoretti et al.
2019 (PRIMAL) · Tang, Berto & Park (EPH) · Stern et al. 2019 (MAPF benchmarks)
# sih_breakingbugs
