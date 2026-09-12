# Dead code inventory

Static sweep plus runtime verification — a symbol is only listed as dead here
if a real 180 s, 4-robot run confirmed it never executes or never varies.

Two kinds of dead code appear below and they need different decisions:

* **Abandoned** — written, superseded, never removed. Delete it.
* **Reserved** — a contract or hook intended for ROS 2 / the comms workstream
  that nothing fills yet. Keep it, but *say so in the code*, because an
  unmarked reserved symbol is indistinguishable from an abandoned one, and
  this repo has repeatedly shipped claims backed by symbols in this state.

---

## Tier 1 — fully dead: zero references anywhere, including tests

| symbol | location | verdict |
|---|---|---|
| `Telemetry` | `amr_msgs.py:270` | **reserved** — its own docstring says *"Dashboard team: build against this and nothing else"*, and the dashboard builds an ad-hoc dict instead. Either adopt it or drop the instruction. |
| `WaitFor` | `amr_msgs.py:256` | **reserved** — the missing input to distributed deadlock resolution |
| `Coordination` | `amr_msgs.py:243` | **reserved** — speed-negotiation message |
| `Obstacle` | `amr_msgs.py:222` | **reserved** — ROS 2 perception |
| `Reroute` | `amr_msgs.py:233` | **reserved** |
| `Reservation.overlaps` | `amr_msgs.py:145` | **abandoned** — `planner.py:69` reimplements the identical interval test inline |
| `is_higher_priority` | `amr_msgs.py:371` | **abandoned** |
| `resolve_by_priority` | `coordination.py:217` | **abandoned** — inlined twice at `sim2d.py:454,481` |

## Tier 2 — test-only: production never calls them

| symbol | location |
|---|---|
| `HybridCoordinator` + `.choose` | `learned.py:204,221` |
| `DeadlockDetector.update_stall` | `coordination.py:239` |
| `Header.is_stale` | `amr_msgs.py:109` — **TTL is therefore never enforced at runtime** |
| `LinkPacket.from_json` | `amr_msgs.py:314` |

A green unit test on a symbol production never calls is the exact pattern
that kept the auction, `WaitFor`, `DeadZone` and the policy dead while the
suite passed.

## Tier 3 — a dead parameter that disables a headline feature

`use_policy` → `Simulation.__init__` → `flags["policy"]`, **never read**.
Only `flags["congestion"]` and `flags["speed_adapt"]` are consumed. Running
seed 3 with the flag on and off gives byte-identical behavioural metrics.
The learned policy is not merely unused — it is unreachable.

## Tier 4 — message fields that never leave their default

Measured across a full run:

| message | field | always |
|---|---|---|
| `RobotState` | `omega` | `0.0` |
| `RobotState` | `lidar_ok`, `comms_ok`, `motors_ok` | `True` |
| `Intent` | `eta_goal`, `next_wx`, `next_wy` | `0.0` |
| `Bid` | `eta` | `0.0` |

Peers receive these fields and can only conclude "nominal". A health flag that
is structurally incapable of going false is worse than no flag — a consumer
will trust it. (`RobotState.capacity_kg` is also constant, but legitimately:
the fleet is homogeneous.)

## Tier 5 — unreachable enum members, and the branches they gate

`RobotMode` is assigned in only 8 places, and **never** to:

```
LOADED · BLOCKED · DEGRADED · CHARGING · FAULT
```

Consequences in live code:

* `dashboard_server.py:165` — *"arrived at pickup and loaded cargo"* — **can
  never fire** (`LOADED` is never set).
* `dashboard_server.py:169` — *"motion paused: path obstructed"* — **can never
  fire** (`BLOCKED` is never set).
* `dashboard_server.py:167` — the guard `prev_mode != RobotMode.LOADED` is
  **vacuous**: `prev_mode` can never be `LOADED`.
* `DEGRADED` duplicates the `Robot.degraded` bool, and the two disagree: the
  bool flips, the mode never does.
* `CHARGING` / `FAULT` are unreachable because no charging or fault path
  exists (see `SECURITY_AUDIT.md` V7).

`MsgType` never sent: `COORDINATION`, `OBSTACLE`, `REROUTE`, `TELEMETRY`,
`WAIT_FOR` — and **`TASK`**, which is special: tasks *are* delivered, but by
writing into `comms._inbox` directly, bypassing the mediator. It shows up as
"never sent" because it never goes through `send()`.

Enums entirely unused: `RerouteReason`, `Resolution`, `ObstacleKind`.
`ReleaseReason`: only `BLOCKED` and `PREEMPTED` are reachable.

## Tier 6 — unused imports (safe to delete)

```
coordination.py        Reservation
run_sih_layout.py      JUNCTIONS, ROBOT_OVERALL_W
dashboard_server.py    Header, INF_COST, Intent, Release, RobotState,
                       WarehouseMap, DeadlockDetector, adapt_speed, math,
                       resolve_auction
```

`resolve_auction` became unused when Scenario 1 was rewritten to observe the
real decentralized auction instead of recomputing it centrally.

## Clean

* **Frontend: 0 dead functions of 41** across `app.js`, `renderer.js` and
  `index.html`. No orphaned handlers.
* `rl/` — self-contained, every symbol reachable from `train.py` / `run_study.py`.

## Inverse gap — live backend, no UI

`add_dead_zone` and `clear_dead_zones` are dispatched by the server and
absent from the frontend (0 references). Every other action is wired. The
feature is reachable only by `curl`, so from an operator's seat it is dead.

---

## Suggested disposition

1. **Delete now, zero risk** — Tier 6 imports, plus `is_higher_priority`,
   `Reservation.overlaps` and `resolve_by_priority` (Tier 1 abandoned).
2. **Mark as reserved** — the Tier 1 message types, with a one-line comment
   naming what will fill them and when. Removes the ambiguity permanently.
3. **Fix or delete the unreachable branches** — the two dashboard log
   messages are user-visible features that cannot occur. Either set `LOADED`
   and `BLOCKED` where they belong, or remove the branches.
4. **Populate the health fields** — `omega`, `lidar_ok`, `comms_ok`,
   `motors_ok`. Already specified in `COMMS_INTERFACE.md` §2.6.
5. **Decide on the policy** (Tier 3) — wire it or state plainly that it is
   trained but not in the control loop.
