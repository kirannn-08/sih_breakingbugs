/**
 * AMR Fleet Mission Control Frontend Application
 * Handles WebSocket communication, UI reactivity, fault injection,
 * task broadcasting, benchmark visualization, and scenario management.
 */

class AppController {
  constructor() {
    this.renderer = new WarehouseRenderer("simCanvas");
    this.socket = null;
    this.state = null;
    this.mapData = null;
    this.activeLogFilter = "ALL";
    this.reconnectTimer = null;

    this.initUI();
    this.connectWebSocket();
  }

  // =========================================================================
  // WEBSOCKET LIFECYCLE
  // =========================================================================

  connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    this.updateConnectionStatus(false, "Connecting...");

    // "local" runs the Python simulation in this tab (static hosting, one
    // simulation per visitor). "server" talks to dashboard_server.py. Both
    // speak the identical message vocabulary, so only the transport differs.
    if (window.AMR_BACKEND === "local") {
      this.socket = new window.LocalSimTransport(
        (msg) => this.updateConnectionStatus(false, msg)
      );
      this.socket.onopen = () => this.updateConnectionStatus(true, "LOCAL SIM");
      this.socket.onmessage = (event) => this.handleTransportMessage(event);
      return;
    }

    try {
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        this.updateConnectionStatus(true, "CONNECTED");
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      this.socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "init") {
            this.mapData = msg.map;
            this.renderer.setMap(msg.map);
            this.populateTaskNodeOptions(msg.map.nodes);
            this.handleStateUpdate(msg.state);
          } else if (msg.type === "state") {
            this.handleStateUpdate(msg.state);
          } else if (msg.type === "scenario_result") {
            this.handleScenarioResult(msg.data);
          }
        } catch (e) {
          console.error("Failed to parse websocket message", e);
        }
      };

      this.socket.onclose = () => {
        this.updateConnectionStatus(false, "DISCONNECTED (Retrying...)");
        if (!this.reconnectTimer) {
          this.reconnectTimer = setTimeout(() => this.connectWebSocket(), 2000);
        }
      };

      this.socket.onerror = (err) => {
        console.error("WebSocket error:", err);
      };
    } catch (e) {
      this.updateConnectionStatus(false, "OFFLINE");
      if (!this.reconnectTimer) {
        this.reconnectTimer = setTimeout(() => this.connectWebSocket(), 3000);
      }
    }
  }

  handleTransportMessage(event) {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "init") {
        this.mapData = msg.map;
        this.renderer.setMap(msg.map);
        this.populateTaskNodeOptions(msg.map.nodes);
        this.handleStateUpdate(msg.state);
      } else if (msg.type === "state") {
        this.handleStateUpdate(msg.state);
      } else if (msg.type === "scenario_result") {
        this.handleScenarioResult(msg.data);
      }
    } catch (e) {
      console.error("Failed to parse simulation message", e);
    }
  }

  send(cmd) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(cmd));
    }
  }

  updateConnectionStatus(online, text) {
    const dot = document.getElementById("connDot");
    const label = document.getElementById("connLabel");
    if (dot && label) {
      dot.className = `status-dot ${online ? "" : "offline"}`;
      label.textContent = text;
    }
  }

  // =========================================================================
  // STATE & TELEMETRY HANDLING
  // =========================================================================

  handleStateUpdate(state) {
    if (!state) return;
    this.state = state;
    this.renderer.updateState(state);

    // Update Clock & Status
    const clockEl = document.getElementById("simClock");
    if (clockEl) {
      const s = state.sim_time || 0.0;
      const mins = Math.floor(s / 60);
      const secs = (s % 60).toFixed(1).padStart(4, "0");
      clockEl.textContent = `T = ${String(mins).padStart(2, "0")}:${secs}s`;
    }

    // Server status indicator
    const serverDot = document.getElementById("serverDot");
    const serverLabel = document.getElementById("serverLabel");
    if (serverDot && serverLabel) {
      serverDot.className = `status-dot ${state.server_online ? "" : "offline"}`;
      serverLabel.textContent = state.server_online ? "SERVER ONLINE" : "SERVER KILLED";
    }

    // Play/Pause button text & state
    const playBtn = document.getElementById("btnPlay");
    if (playBtn) {
      playBtn.textContent = state.is_running ? "⏸ Pause" : "▶ Play";
      playBtn.className = state.is_running ? "btn btn-danger" : "btn btn-success";
    }

    // Auto-tasks checkbox
    const autoTasksBox = document.getElementById("autoTasksCheck");
    if (autoTasksBox) autoTasksBox.checked = state.auto_tasks;

    // Loss slider
    const lossSlider = document.getElementById("lossSlider");
    const lossVal = document.getElementById("lossVal");
    if (lossSlider && lossVal) {
      const pct = Math.round((state.comms?.loss_rate || 0) * 100);
      lossSlider.value = pct;
      lossVal.textContent = `${pct}%`;
    }

    // Render Subcomponents
    this.renderRobotCards(state.robots);
    this.renderCommsMatrix(state.comms);
    this.renderTaskBoard(state.tasks);
    this.renderBenchmarks(state.metrics);
    this.renderEvents(state.events);
  }

  // =========================================================================
  // SUBCOMPONENT RENDERING
  // =========================================================================

  renderRobotCards(robots) {
    const container = document.getElementById("robotCardsContainer");
    if (!container || !robots) return;

    container.innerHTML = robots
      .map((r) => {
        const vPct = Math.min(100, (r.v / (r.v_nom || 0.6)) * 100);
        const batPct = Math.round(r.battery * 100);
        const batClass = batPct < 30 ? "danger" : batPct < 60 ? "warning" : "";

        const peersText = r.peers_heard && r.peers_heard.length > 0
          ? r.peers_heard.map((id) => `AMR ${id}`).join(", ")
          : "None (Isolated)";

        const goalText = r.goal ? `(${r.goal[0]}, ${r.goal[1]})` : "None";
        const taskText = r.task_id ? `#${r.task_id} [${r.phase}]` : "None";

        return `
          <div class="robot-card">
            <div class="robot-header">
              <div class="robot-id-badge" style="color: ${this.renderer.robotColors[r.id]?.main || '#0284c7'}">
                <span>🤖 AMR ${r.id}</span>
                ${r.degraded ? '<span class="badge-tag" style="background: #f3e8ff; color:#7e22ce; border-color:#d8b4fe;">DEGRADED</span>' : ''}
                ${r.yielding ? '<span class="badge-tag" style="background: #fef3c7; color:#d97706; border-color:#fde68a;">YIELDING</span>' : ''}
              </div>
              <span class="mode-badge mode-${r.mode}">${r.mode}</span>
            </div>

            <div class="robot-grid">
              <div class="robot-prop"><span>Battery:</span> <span class="robot-val">${batPct}%</span></div>
              <div class="robot-prop"><span>Speed:</span> <span class="robot-val">${r.v.toFixed(2)} m/s</span></div>
              <div class="robot-prop"><span>Task:</span> <span class="robot-val">${taskText}</span></div>
              <div class="robot-prop"><span>Goal:</span> <span class="robot-val">${goalText}</span></div>
              <div class="robot-prop"><span>Distance:</span> <span class="robot-val">${r.dist.toFixed(1)} m</span></div>
              <div class="robot-prop"><span>Stops:</span> <span class="robot-val">${r.stops}</span></div>
            </div>

            <!-- Speed & Battery Meters -->
            <div style="margin-bottom: 6px;">
              <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim);">
                <span>Battery Level</span>
                <span>Velocity (${r.v.toFixed(2)} / ${r.v_nom.toFixed(2)} m/s)</span>
              </div>
              <div style="display: flex; gap: 8px;">
                <div class="meter-bar" style="flex: 1;"><div class="meter-fill ${batClass}" style="width: ${batPct}%;"></div></div>
                <div class="meter-bar" style="flex: 1;"><div class="meter-fill" style="width: ${vPct}%; background: ${this.renderer.robotColors[r.id]?.main || '#0284c7'};"></div></div>
              </div>
            </div>

            <div style="font-size: 11px; color: var(--text-muted); display: flex; justify-content: space-between;">
              <span>Peers Heard: <strong style="color: var(--text-main);">${peersText}</strong></span>
              <span>Tasks Done: <strong style="color: var(--accent-green);">${r.task_id ? 1 : 0}</strong></span>
            </div>
          </div>
        `;
      })
      .join("");
  }

  renderCommsMatrix(comms) {
    const table = document.getElementById("commsMatrixTable");
    if (!table || !comms || !this.state?.robots) return;

    const robots = this.state.robots;
    const n = robots.length;
    const links = comms.link_matrix || {};

    let html = "<thead><tr><th></th>";
    for (let j = 1; j <= n; j++) {
      html += `<th>AMR ${j}</th>`;
    }
    html += "</tr></thead><tbody>";

    for (let i = 1; i <= n; i++) {
      html += `<tr><th>AMR ${i}</th>`;
      for (let j = 1; j <= n; j++) {
        if (i === j) {
          html += `<td class="matrix-cell self">—</td>`;
        } else {
          const key = `${i}->${j}`;
          const isUp = links[key] !== false;
          const cls = isUp ? "connected" : "severed";
          const label = isUp ? "ONLINE" : "CUT";
          html += `
            <td class="matrix-cell ${cls}" onclick="window.app.toggleLink(${i}, ${j}, ${isUp})" title="Click to ${isUp ? 'Cut' : 'Restore'} link ${i} <-> ${j}">
              ${label}
            </td>
          `;
        }
      }
      html += "</tr>";
    }
    html += "</tbody>";
    table.innerHTML = html;

    // Network stats
    const stats = comms.stats || {};
    document.getElementById("statSent").textContent = stats.sent || 0;
    document.getElementById("statDelivered").textContent = stats.delivered || 0;
    document.getElementById("statDropped").textContent = stats.dropped || 0;
    document.getElementById("statBandwidth").textContent = `${Math.round((stats.bytes || 0) / 1024)} KB`;
    document.getElementById("statLoss").textContent = `${stats.loss_pct || 0}%`;
  }

  renderTaskBoard(tasks) {
    if (!tasks) return;

    const openContainer = document.getElementById("openTasksList");
    if (openContainer) {
      if (!tasks.open || tasks.open.length === 0) {
        openContainer.innerHTML = '<div style="color: var(--text-dim); font-size: 11px; padding: 6px;">No open unassigned tasks.</div>';
      } else {
        openContainer.innerHTML = tasks.open
          .map(
            (t) => `
          <div class="task-card" style="border-left-color: var(--accent-amber);">
            <div style="display: flex; justify-content: space-between; font-weight: 700;">
              <span>Task #${t.task_id} (AUCTION PENDING)</span>
              <span class="badge-tag">P${t.priority}</span>
            </div>
            <div style="color: var(--text-muted); margin-top: 4px;">
              Pickup: (${t.pickup[0]}, ${t.pickup[1]}) ➔ Dropoff: (${t.dropoff[0]}, ${t.dropoff[1]}) [${t.payload_kg} kg]
            </div>
          </div>
        `
          )
          .join("");
      }
    }

    const activeContainer = document.getElementById("activeTasksList");
    if (activeContainer) {
      if (!tasks.active || tasks.active.length === 0) {
        activeContainer.innerHTML = '<div style="color: var(--text-dim); font-size: 11px; padding: 6px;">No robots currently carrying tasks.</div>';
      } else {
        activeContainer.innerHTML = tasks.active
          .map(
            (t) => `
          <div class="task-card" style="border-left-color: var(--accent-green);">
            <div style="display: flex; justify-content: space-between; font-weight: 700;">
              <span>Task #${t.task_id} ➔ AMR ${t.robot_id}</span>
              <span class="badge-tag" style="background: rgba(16,185,129,0.15); color: #34d399;">${t.phase}</span>
            </div>
            <div style="color: var(--text-muted); margin-top: 4px;">
              Route: (${t.pickup[0]}, ${t.pickup[1]}) ➔ (${t.dropoff[0]}, ${t.dropoff[1]}) [${t.payload_kg} kg]
            </div>
          </div>
        `
          )
          .join("");
      }
    }

    const completedCountEl = document.getElementById("kpiCompletedTasks");
    if (completedCountEl) completedCountEl.textContent = tasks.completed_count || 0;
  }

  renderBenchmarks(metrics) {
    if (!metrics) return;

    const setVal = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    setVal("kpiThroughput", `${metrics.throughput_per_min || 0}/min`);
    setVal("kpiAvgTime", `${metrics.avg_task_time || 0}s`);
    setVal("kpiCollisions", metrics.collisions || 0);
    setVal("kpiNearMisses", metrics.near_misses || 0);
    setVal("kpiDeadlocks", metrics.deadlocks || 0);
    setVal("kpiFullStops", metrics.full_stops || 0);
  }

  renderEvents(events) {
    const stream = document.getElementById("logStream");
    if (!stream || !events) return;

    const filtered =
      this.activeLogFilter === "ALL"
        ? events
        : events.filter((e) => e.category === this.activeLogFilter);

    // Keep scroll position if user scrolled up
    const isAtBottom = stream.scrollHeight - stream.scrollTop <= stream.clientHeight + 20;

    stream.innerHTML = filtered
      .map(
        (e) => `
      <div class="log-row">
        <span class="log-time">[T=${e.sim_time.toFixed(1)}s]</span>
        <span class="log-cat cat-${e.category}">${e.category}</span>
        <span class="log-msg ${e.level}">${e.message}</span>
      </div>
    `
      )
      .join("");

    if (isAtBottom) {
      stream.scrollTop = stream.scrollHeight;
    }
  }

  // =========================================================================
  // ACTIONS & CONTROLS DISPATCH
  // =========================================================================

  togglePlay() {
    const isRunning = this.state?.is_running;
    this.send({ action: isRunning ? "pause" : "play" });
  }

  stepSim() {
    this.send({ action: "step" });
  }

  resetSim() {
    this.send({ action: "reset" });
  }

  setSpeed(val) {
    this.send({ action: "set_speed", value: parseFloat(val) });
  }

  toggleAutoTasks(enabled) {
    this.send({ action: "set_auto_tasks", enabled: Boolean(enabled) });
  }

  toggleLink(a, b, isCurrentlyUp) {
    if (isCurrentlyUp) {
      this.send({ action: "cut_link", a, b });
    } else {
      this.send({ action: "restore_link", a, b });
    }
  }

  isolateRobot(robotId) {
    this.send({ action: "isolate", robot_id: robotId });
  }

  cutAllLinks() {
    this.send({ action: "cut_all" });
  }

  restoreAllLinks() {
    this.send({ action: "restore_all" });
  }

  setLossRate(rate) {
    this.send({ action: "set_loss_rate", rate: parseFloat(rate) });
  }

  toggleServerOnline() {
    const online = this.state?.server_online;
    if (online) {
      this.send({ action: "kill_server" });
    } else {
      this.send({ action: "restore_server" });
    }
  }

  broadcastCustomTask() {
    const pickEl = document.getElementById("taskPickupSelect");
    const dropEl = document.getElementById("taskDropoffSelect");
    const weightEl = document.getElementById("taskWeightInput");
    const prioEl = document.getElementById("taskPrioritySelect");

    if (!pickEl || !dropEl) return;

    const [px, py] = pickEl.value.split(",").map(Number);
    const [dx, dy] = dropEl.value.split(",").map(Number);
    const weight = parseFloat(weightEl?.value || "5.0");
    const prio = parseInt(prioEl?.value || "1");

    this.send({
      action: "broadcast_task",
      pickup: [px, py],
      dropoff: [dx, dy],
      payload_kg: weight,
      priority: prio,
    });
  }

  runScenario(id) {
    this.send({ action: "run_scenario", scenario_id: parseInt(id) });
  }

  handleScenarioResult(res) {
    if (res.message) {
      alert(`Scenario ${res.scenario} Triggered:\n${res.message}`);
    }
  }

  // =========================================================================
  // UI INITIALIZATION
  // =========================================================================

  initUI() {
    // Tabs Navigation
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
        btn.classList.add("active");
        const target = document.getElementById(btn.dataset.tab);
        if (target) target.classList.add("active");
      });
    });

    // Layer toggles
    const bindToggle = (id, prop) => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("change", (e) => {
          this.renderer[prop] = e.target.checked;
          this.renderer.requestRender();
        });
      }
    };

    bindToggle("chkRobots", "showRobots");
    bindToggle("chkLidar", "showLidar");
    bindToggle("chkPaths", "showPaths");
    bindToggle("chkReservations", "showReservations");
    bindToggle("chkBays", "showBays");
    bindToggle("chkLabels", "showLabels");
    bindToggle("chkTrails", "showTrails");

    // Event Log Filter Pills
    document.querySelectorAll(".filter-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        document.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        this.activeLogFilter = pill.dataset.filter;
        if (this.state?.events) this.renderEvents(this.state.events);
      });
    });
  }

  populateTaskNodeOptions(nodes) {
    const pickEl = document.getElementById("taskPickupSelect");
    const dropEl = document.getElementById("taskDropoffSelect");
    if (!pickEl || !dropEl || !nodes) return;

    pickEl.innerHTML = "";
    dropEl.innerHTML = "";

    for (const [name, n] of Object.entries(nodes)) {
      if (n.kind === "pickup") {
        pickEl.innerHTML += `<option value="${n.cx},${n.cy}">${name} (${n.cx}, ${n.cy})</option>`;
      } else if (n.kind === "dropoff") {
        dropEl.innerHTML += `<option value="${n.cx},${n.cy}">${name} (${n.cx}, ${n.cy})</option>`;
      }
    }
  }
}

// Global initialization
window.addEventListener("DOMContentLoaded", () => {
  window.app = new AppController();
});

