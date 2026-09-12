"""
Tests for dashboard server simulation manager and API models.
"""

import json
from dashboard_server import SimulationManager


def test_simulation_manager_init():
    sm = SimulationManager(n_robots=4, seed=1)
    assert sm.n_robots == 4
    assert len(sm.sim.robots) == 4
    assert sm.is_running is False


def test_simulation_manager_map_data():
    sm = SimulationManager(n_robots=4, seed=1)
    m = sm.get_map_data()
    assert m["w"] == 40
    assert m["h"] == 30
    assert "nodes" in m
    assert len(m["nodes"]) >= 9
    assert len(m["passing_bays"]) >= 10
    # Must be JSON-serializable
    json.dumps(m)


def test_simulation_manager_state_data():
    sm = SimulationManager(n_robots=4, seed=1)
    state = sm.get_state()
    assert "robots" in state
    assert len(state["robots"]) == 4
    assert "comms" in state
    assert "link_matrix" in state["comms"]
    assert "tasks" in state
    assert "metrics" in state
    # Must be JSON-serializable
    json.dumps(state)


def test_simulation_manager_fault_injection():
    sm = SimulationManager(n_robots=4, seed=1)
    sm.cut_link(1, 2)
    assert sm.sim.comms.link_matrix[(1, 2)] is False
    assert sm.sim.comms.link_matrix[(2, 1)] is False

    sm.restore_link(1, 2)
    assert sm.sim.comms.link_matrix[(1, 2)] is True

    sm.isolate_robot(1)
    for other in [2, 3, 4]:
        assert sm.sim.comms.link_matrix[(1, other)] is False

    sm.restore_all_links()
    for (i, j), active in sm.sim.comms.link_matrix.items():
        assert active is True

    sm.cut_all_links()
    for (i, j), active in sm.sim.comms.link_matrix.items():
        assert active is False

    sm.restore_all_links()


def test_simulation_manager_step_and_task():
    sm = SimulationManager(n_robots=4, seed=1)
    t = sm.broadcast_task(8, 9, 2, 27, payload_kg=5.0, priority_class=1)
    assert t is not None
    assert t.task_id in sm.sim.open_tasks
    for _ in range(5):
        sm.step()
    state = sm.get_state()
    assert state["sim_time"] > 0.0


def test_simulation_manager_scenarios():
    sm = SimulationManager(n_robots=4, seed=1)
    for s_id in [1, 2, 3, 4, 5, 6, 7]:
        res = sm.run_scenario(s_id)
        assert res["status"] == "ok"

