import json
import itertools
from sim2d import Simulation
from amr_msgs import Task, make_header

class AuditSimulation(Simulation):
    def __init__(self, n_robots, custom_tasks):
        self.custom_tasks = custom_tasks
        self.task_idx = 0
        super().__init__(n_robots=n_robots, seed=1)

    def announce_task(self) -> None:
        if self.task_idx >= len(self.custom_tasks):
            return
        
        task_info = self.custom_tasks[self.task_idx]
        self.task_idx += 1
        
        pick_name = task_info["pickup"]
        drop_name = task_info["dropoff"]
        priority_class = task_info.get("priority", 1)
        payload = task_info.get("payload", 5.0)
        
        nodes = self.wmap.nodes
        if pick_name not in nodes or drop_name not in nodes:
            return
            
        pick = nodes[pick_name]
        drop = nodes[drop_name]
        
        t = Task(header=make_header(0, self.t), task_id=self.next_task_id,
                 pickup_cx=pick.cx, pickup_cy=pick.cy,
                 dropoff_cx=drop.cx, dropoff_cy=drop.cy,
                 payload_kg=payload,
                 priority_class=priority_class,
                 announced_at=self.t)
        
        self.next_task_id += 1
        self.open_tasks[t.task_id] = t
        self.task_announced_at[t.task_id] = self.t
        for r in self.robots:
            from amr_msgs import wrap, MsgType
            self.comms._inbox[r.id].append(
                wrap(t, MsgType.TASK, 0, r.id, self.t))


def run_all_cases():
    # Define possible values for permutation
    amr_counts = [2, 3, 4]
    pickups = ["P1", "P2", "P3", "P4"]
    dropoffs = ["D1", "D2", "D3"]
    payloads = [2.0, 10.0, 25.0]  # 25 is > capacity (20) to check how it handles it
    priorities = [1, 5]
    
    # We will construct a few interesting overlapping combinations
    scenarios = [
        # Scenario 1: cross traffic
        [
            {"pickup": "P1", "dropoff": "D3", "payload": 5.0, "priority": 1},
            {"pickup": "P3", "dropoff": "D1", "payload": 5.0, "priority": 1},
            {"pickup": "P2", "dropoff": "D2", "payload": 5.0, "priority": 1},
            {"pickup": "P4", "dropoff": "D2", "payload": 5.0, "priority": 1}
        ],
        # Scenario 2: all want the same dropoff (congestion)
        [
            {"pickup": "P1", "dropoff": "D1", "payload": 5.0, "priority": 5},
            {"pickup": "P2", "dropoff": "D1", "payload": 5.0, "priority": 1},
            {"pickup": "P3", "dropoff": "D1", "payload": 5.0, "priority": 1},
            {"pickup": "P4", "dropoff": "D1", "payload": 5.0, "priority": 5}
        ],
        # Scenario 3: Heavy payloads and varied priority
        [
            {"pickup": "P1", "dropoff": "D3", "payload": 19.0, "priority": 1},
            {"pickup": "P4", "dropoff": "D1", "payload": 10.0, "priority": 5},
            {"pickup": "P2", "dropoff": "D2", "payload": 2.0, "priority": 1},
            {"pickup": "P3", "dropoff": "D2", "payload": 15.0, "priority": 5}
        ],
        # Scenario 4: Empty tasks (forces robots to go to charging bays immediately, exposing bay_taken bug)
        []
    ]

    all_bugs = []
    
    print("Running simulations...")
    for n_robots in amr_counts:
        for s_idx, scenario in enumerate(scenarios):
            tasks_to_run = scenario[:n_robots]
            sim = AuditSimulation(n_robots=n_robots, custom_tasks=tasks_to_run)
            
            # Step manually to check charging state at each tick
            try:
                max_duration = 180.0
                next_task = 1.0
                task_interval = 2.0
                max_tasks = len(tasks_to_run)
                
                sim.t = 0.0
                
                # Check for bugs tick by tick
                charging_bug_detected = False
                near_miss_bug_detected = False
                
                while sim.t < max_duration:
                    if sim.t >= next_task and sim.next_task_id <= max_tasks:
                        sim.announce_task()
                        next_task = sim.t + task_interval
                    
                    sim.step()
                    
                    # Audit logic 1: Charging stations updating
                    # Check if a robot says it's charging but not at a bay
                    for r in sim.robots:
                        if r.charging and not r.at_bay():
                            all_bugs.append(f"Bug [Charging]: AMR {r.id} charging at {r.cell} which is not a charger bay! (t={sim.t})")
                            charging_bug_detected = True
                            
                        # Bug in coordination/sim2d: a robot wants to charge but the nearest_free_bay returns a bay that IS taken 
                        # because bay_taken check fails (e.g., peer is slightly outside 1.2m or path ended but robot still there).
                        if r.mode.name == "CHARGING" and r.goal is not None and r.goal == r.cell:
                            # It is at its goal, is another robot there?
                            for o in sim.robots:
                                if o.id != r.id and o.cell == r.cell:
                                    all_bugs.append(f"Bug [Charging]: AMR {r.id} and AMR {o.id} overlapping at charger bay {r.cell}! (t={sim.t})")
                                    charging_bug_detected = True
                    
                    if max_tasks > 0 and sim.metrics.tasks_completed >= max_tasks and not sim.open_tasks:
                        break
                    elif max_tasks == 0 and sim.t > 30.0:
                        break
                        
                sim.aggregate_metrics()
                
                # Audit logic 2: Near misses
                # If there are near misses but collisions = 0, is it normal? Yes, but if collisions > 0 it's a bug.
                if sim.metrics.collisions > 0:
                    all_bugs.append(f"Bug [Collision]: {sim.metrics.collisions} collisions occurred! (n_robots={n_robots}, scenario={s_idx})")
                
                if sim.metrics.near_misses > 0:
                    all_bugs.append(f"Finding [Near Miss]: {sim.metrics.near_misses} near misses occurred. (n_robots={n_robots}, scenario={s_idx})")
                    
            except Exception as e:
                all_bugs.append(f"Bug [Crash]: Simulation crashed for n_robots={n_robots}, scenario={s_idx} with error: {e}")

    # Deduplicate bugs
    unique_bugs = list(set(all_bugs))
    
    explicit_bugs = [
        "",
        "=== EXPLICIT CODE BUGS IDENTIFIED ===",
        "1. CHARGING STATION OCCUPATION (bay_taken tie-breaker flaw in sim2d.py):",
        "   In `sim2d.py -> bay_taken()`, the tie-breaker logic is:",
        "   `if it.path_cells and tuple(it.path_cells[-1]) == bay: if pid < self.id: return True`",
        "   This means if peer `pid` is heading to the bay, we only consider it 'taken' if `pid < self.id`.",
        "   If `pid > self.id`, `bay_taken` returns False, leading BOTH robots to head to the same bay.",
        "   Fix: The tie-break must be symmetrical, or there should be a proper claim protocol for bays.",
        "",
        "2. NEAR MISSES (lidar line_of_sight braking bypass):",
        "   In `sim2d.py -> step()`, the LiDAR loop skips braking for peers within `D_NEAR` if they are occluded.",
        "   Because a rack corner can hide a crossing lane, robots approach at `v_nom` and suddenly see",
        "   each other inside `D_NEAR`. The brake logic applies, but only after they are already dangerously close,",
        "   leading to numerous near-misses being recorded at blind corners.",
        "   Fix: AMRs must slow down prophylactically when approaching blind corners (`is_blind_corner`)."
    ]
    
    with open("audit_findings.txt", "w") as f:
        for b in unique_bugs:
            f.write(b + "\n")
            print(b)
        for eb in explicit_bugs:
            f.write(eb + "\n")
            print(eb)
            
    print("Audit finished. Findings saved to audit_findings.txt.")

if __name__ == "__main__":
    run_all_cases()
