/**
 * Runs the real Python simulation in the browser via Pyodide.
 *
 * Why: static hosting (Netlify) cannot run tornado, and every visitor must get
 * their own simulation that resets on refresh. Running sim_manager.py in the
 * tab gives both for free -- there is no shared server state to isolate.
 *
 * LocalSimTransport deliberately mimics the WebSocket surface app.js already
 * uses (readyState / onopen / onmessage / send / close) so the frontend keeps
 * one code path whether it is talking to tornado or to the tab itself.
 */

const PY_MODULES = [
  "amr_msgs.py", "comms.py", "coordination.py", "features.py", "learned.py",
  "perception.py", "planner.py", "sim2d.py", "sim_manager.py", "warehouse_map.py",
];

// Mirrors dashboard_server.sim_tick(): same 10 Hz base rate, same speed handling.
const PY_DRIVER = `
import json
from sim_manager import SimulationManager, dispatch

_mgr = SimulationManager()

def init_msg():
    return json.dumps({"type": "init", "map": _mgr.get_map_data(),
                       "state": _mgr.get_state()})

def _state_msg():
    return json.dumps({"type": "state", "state": _mgr.get_state()})

def tick():
    if not _mgr.is_running:
        return None
    for _ in range(max(1, int(_mgr.speed_multiplier))):
        _mgr.step()
    return _state_msg()

def handle(raw):
    out = []
    try:
        res = dispatch(_mgr, json.loads(raw))
        if res.get("type") == "scenario_result":
            out.append(json.dumps(res))
    except Exception as exc:
        out.append(json.dumps({"type": "error", "error": str(exc)}))
    out.append(_state_msg())
    return json.dumps(out)
`;

class LocalSimTransport {
  constructor(statusFn) {
    this.readyState = 0; // CONNECTING
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
    this._timer = null;
    this._py = null;
    this._status = statusFn || (() => {});
    this._boot();
  }

  async _boot() {
    try {
      this._status("Loading Python runtime...");
      const py = await loadPyodide();

      this._status("Loading numpy...");
      await py.loadPackage("numpy");

      this._status("Loading simulation...");
      const base = new URL("py/", window.location.href).href;
      const sources = await Promise.all(
        PY_MODULES.map(async (name) => {
          const r = await fetch(base + name);
          if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
          return [name, await r.text()];
        })
      );
      for (const [name, src] of sources) py.FS.writeFile("/home/pyodide/" + name, src);
      py.runPython('import sys; sys.path.insert(0, "/home/pyodide")');
      py.runPython(PY_DRIVER);

      this._py = py;
      this.readyState = 1; // OPEN
      if (this.onopen) this.onopen();

      this._emit(py.globals.get("init_msg")());
      const tick = py.globals.get("tick");
      this._timer = setInterval(() => {
        if (this.readyState !== 1) return;
        const msg = tick();
        if (msg) this._emit(msg);
      }, 100);
    } catch (e) {
      console.error("Local simulation failed to start:", e);
      this.readyState = 3; // CLOSED
      this._status("SIM FAILED TO LOAD");
      if (this.onerror) this.onerror(e);
    }
  }

  _emit(data) {
    if (this.onmessage) this.onmessage({ data });
  }

  send(raw) {
    if (this.readyState !== 1 || !this._py) return;
    const out = JSON.parse(this._py.globals.get("handle")(raw));
    for (const msg of out) this._emit(msg);
  }

  close() {
    this.readyState = 3;
    if (this._timer) clearInterval(this._timer);
    if (this.onclose) this.onclose();
  }
}

window.LocalSimTransport = LocalSimTransport;
