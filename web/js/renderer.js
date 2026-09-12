/**
 * 2D Warehouse Visualizer & High-DPI Canvas Renderer
 * Clean white background with crisp grey accents and vivid multi-color AMRs, paths, and stations.
 */

class WarehouseRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");

    this.mapData = null;
    this.state = null;

    // Viewport transforms (Pan & Zoom)
    this.scale = 24.0; // pixels per metre
    this.offsetX = 40.0;
    this.offsetY = 40.0;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;

    // Layer toggles
    this.showRobots = true;
    this.showLidar = true;
    this.showPaths = true;
    this.showReservations = true;
    this.showBays = true;
    this.showLabels = true;
    this.showTrails = true;

    // Trails history: robot_id -> array of {x, y}
    this.trails = {};

    // Distinct multi-color palette for AMRs and paths
    this.robotColors = {
      1: {
        main: "#0284c7",       // Vivid Sky Blue
        trail: "rgba(2, 132, 199, 0.6)",
        path: "#0284c7",
        glow: "rgba(2, 132, 199, 0.15)",
        dark: "#0369a1",
        bg: "#f0f9ff"
      },
      2: {
        main: "#059669",       // Vivid Emerald
        trail: "rgba(5, 150, 105, 0.6)",
        path: "#059669",
        glow: "rgba(5, 150, 105, 0.15)",
        dark: "#047857",
        bg: "#f0fdf4"
      },
      3: {
        main: "#d97706",       // Vivid Amber
        trail: "rgba(217, 119, 6, 0.6)",
        path: "#d97706",
        glow: "rgba(217, 119, 6, 0.15)",
        dark: "#b45309",
        bg: "#fffbeb"
      },
      4: {
        main: "#e11d48",       // Vivid Rose
        trail: "rgba(225, 29, 72, 0.6)",
        path: "#e11d48",
        glow: "rgba(225, 29, 72, 0.15)",
        dark: "#be123c",
        bg: "#fff1f2"
      },
    };

    this.initEvents();
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  initEvents() {
    this.canvas.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.dragStartX = e.clientX - this.offsetX;
      this.dragStartY = e.clientY - this.offsetY;
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isDragging) return;
      this.offsetX = e.clientX - this.dragStartX;
      this.offsetY = e.clientY - this.dragStartY;
      this.requestRender();
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Zoom toward cursor
      this.offsetX = mouseX - (mouseX - this.offsetX) * zoomFactor;
      this.offsetY = mouseY - (mouseY - this.offsetY) * zoomFactor;
      this.scale *= zoomFactor;
      this.scale = Math.max(8.0, Math.min(80.0, this.scale));
      this.requestRender();
    });
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.width = rect.width;
    this.height = rect.height;
    this.requestRender();
  }

  fitToMap() {
    if (!this.mapData) return;
    const mapW = this.mapData.width_m || 20.0;
    const mapH = this.mapData.height_m || 15.0;

    const scaleX = (this.width - 80) / mapW;
    const scaleY = (this.height - 80) / mapH;
    this.scale = Math.min(scaleX, scaleY);
    this.offsetX = (this.width - mapW * this.scale) / 2;
    this.offsetY = (this.height - mapH * this.scale) / 2;
    this.requestRender();
  }

  setMap(mapData) {
    this.mapData = mapData;
    this.fitToMap();
  }

  updateState(state) {
    this.state = state;

    // Update trails
    if (state && state.robots) {
      for (const r of state.robots) {
        if (!this.trails[r.id]) this.trails[r.id] = [];
        const t = this.trails[r.id];
        if (t.length === 0 || Math.hypot(t[t.length - 1].x - r.x, t[t.length - 1].y - r.y) > 0.18) {
          t.push({ x: r.x, y: r.y });
          if (t.length > 35) t.shift();
        }
      }
    }
    this.requestRender();
  }

  worldToScreen(wx, wy) {
    return {
      x: this.offsetX + wx * this.scale,
      y: this.offsetY + wy * this.scale,
    };
  }

  requestRender() {
    requestAnimationFrame(() => this.render());
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    if (!this.mapData) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "14px monospace";
      ctx.fillText("Connecting to simulation...", 40, 50);
      return;
    }

    const { w, h, cell_size, grid, zone, passing_bays, nodes } = this.mapData;

    // 1. Draw Clean White Floor Base
    ctx.fillStyle = "#ffffff";
    const topLeft = this.worldToScreen(0, 0);
    const mapWidthPx = w * cell_size * this.scale;
    const mapHeightPx = h * cell_size * this.scale;

    // Map shadow
    ctx.shadowColor = "rgba(0, 0, 0, 0.06)";
    ctx.shadowBlur = 16;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 4;
    ctx.fillRect(topLeft.x, topLeft.y, mapWidthPx, mapHeightPx);
    ctx.shadowColor = "transparent";

    // Floor outline border
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(topLeft.x, topLeft.y, mapWidthPx, mapHeightPx);

    // 2. Draw Floor Grid & Warehouse Features
    const csPx = cell_size * this.scale;

    for (let cy = 0; cy < h; cy++) {
      for (let cx = 0; cx < w; cx++) {
        const type = grid[cy][cx];
        const scr = this.worldToScreen(cx * cell_size, cy * cell_size);

        if (type === 1) {
          // WALL (Dark Slate)
          ctx.fillStyle = "#334155";
          ctx.fillRect(scr.x, scr.y, csPx, csPx);
          ctx.strokeStyle = "#1e293b";
          ctx.lineWidth = 1;
          ctx.strokeRect(scr.x, scr.y, csPx, csPx);
        } else if (type === 2) {
          // SHELF / RACK (Light metallic grey with crisp structural borders)
          ctx.fillStyle = "#f1f5f9";
          ctx.fillRect(scr.x, scr.y, csPx, csPx);

          // Rack outer frame
          ctx.strokeStyle = "#94a3b8";
          ctx.lineWidth = 1;
          ctx.strokeRect(scr.x + 0.5, scr.y + 0.5, csPx - 1, csPx - 1);

          // Structural diagonal brace
          ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(scr.x + 2, scr.y + 2);
          ctx.lineTo(scr.x + csPx - 2, scr.y + csPx - 2);
          ctx.stroke();
        } else {
          // Free cell grid
          ctx.strokeStyle = "#f1f5f9";
          ctx.lineWidth = 0.5;
          ctx.strokeRect(scr.x, scr.y, csPx, csPx);

          // Narrow Aisle Subtle Warm Tint
          if (zone && zone[cy][cx] === 0) {
            ctx.fillStyle = "rgba(245, 158, 11, 0.04)";
            ctx.fillRect(scr.x, scr.y, csPx, csPx);
          }
        }
      }
    }

    // 3. Draw Passing Bays (Vivid Cyan Notches)
    if (this.showBays && passing_bays) {
      for (const [bx, by] of passing_bays) {
        const scr = this.worldToScreen(bx * cell_size, by * cell_size);
        ctx.fillStyle = "#e0f2fe"; // Light cyan fill
        ctx.fillRect(scr.x, scr.y, csPx, csPx);

        ctx.strokeStyle = "#0284c7"; // Cyan border
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(scr.x + 1, scr.y + 1, csPx - 2, csPx - 2);
        ctx.setLineDash([]);

        if (this.scale > 18) {
          ctx.fillStyle = "#0284c7";
          ctx.font = "bold 8px monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("BAY", scr.x + csPx / 2, scr.y + csPx / 2);
        }
      }
    }

    // 4. Draw Space-Time Reservations (Multi-Color Overlay)
    if (this.showReservations && this.state && this.state.robots) {
      for (const r of this.state.robots) {
        const color = this.robotColors[r.id] || { main: "#0284c7", glow: "rgba(2, 132, 199, 0.15)" };
        for (const res of r.reservations || []) {
          const scr = this.worldToScreen(res.cx * cell_size, res.cy * cell_size);
          ctx.fillStyle = color.glow;
          ctx.fillRect(scr.x + 1, scr.y + 1, csPx - 2, csPx - 2);
          ctx.strokeStyle = color.main;
          ctx.lineWidth = 1;
          ctx.strokeRect(scr.x + 1, scr.y + 1, csPx - 2, csPx - 2);
        }
      }
    }

    // 5. Draw Warehouse Locations (Pickups, Dropoffs, Chargers)
    if (nodes) {
      for (const [name, n] of Object.entries(nodes)) {
        const scr = this.worldToScreen(n.wx, n.wy);
        const radius = Math.max(7, this.scale * 0.42);

        let color = "#2563eb";
        let iconText = "P";
        let haloColor = "rgba(37, 99, 235, 0.15)";

        if (n.kind === "pickup") {
          color = "#2563eb";       // Royal Blue
          haloColor = "rgba(37, 99, 235, 0.15)";
          iconText = "P";
        } else if (n.kind === "dropoff") {
          color = "#059669";       // Emerald
          haloColor = "rgba(5, 150, 105, 0.15)";
          iconText = "D";
        } else if (n.kind === "charger") {
          color = "#d97706";       // Amber
          haloColor = "rgba(217, 119, 6, 0.15)";
          iconText = "⚡";
        }

        // Location halo
        ctx.beginPath();
        ctx.arc(scr.x, scr.y, radius + 4, 0, Math.PI * 2);
        ctx.fillStyle = haloColor;
        ctx.fill();

        // Location node circle
        ctx.beginPath();
        ctx.arc(scr.x, scr.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Icon text
        ctx.fillStyle = "#ffffff";
        ctx.font = `bold ${Math.max(9, radius * 0.9)}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(iconText, scr.x, scr.y);

        // Location name label
        if (this.showLabels) {
          ctx.fillStyle = "#0f172a";
          ctx.font = "bold 10px monospace";
          ctx.fillText(name, scr.x, scr.y + radius + 11);
        }
      }
    }

    // 6. Draw Robot Trails (Multi-Color)
    if (this.showTrails && this.trails) {
      for (const [rid, points] of Object.entries(this.trails)) {
        if (points.length < 2) continue;
        const color = this.robotColors[rid] || { trail: "rgba(2, 132, 199, 0.6)" };
        ctx.strokeStyle = color.trail;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const scr = this.worldToScreen(points[i].x, points[i].y);
          if (i === 0) ctx.moveTo(scr.x, scr.y);
          else ctx.lineTo(scr.x, scr.y);
        }
        ctx.stroke();
      }
    }

    // 7. Draw Planned Paths (Multi-Color)
    if (this.showPaths && this.state && this.state.robots) {
      for (const r of this.state.robots) {
        if (!r.path || r.path.length <= 1) continue;
        const color = this.robotColors[r.id] || { main: "#0284c7" };
        ctx.strokeStyle = color.main;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();

        const robotScr = this.worldToScreen(r.x, r.y);
        ctx.moveTo(robotScr.x, robotScr.y);

        for (let i = r.path_idx; i < r.path.length; i++) {
          const [cx, cy] = r.path[i];
          const scr = this.worldToScreen((cx + 0.5) * cell_size, (cy + 0.5) * cell_size);
          ctx.lineTo(scr.x, scr.y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // 8. Draw Robots & Safety Zones (Multi-Color)
    if (this.showRobots && this.state && this.state.robots) {
      for (const r of this.state.robots) {
        this.drawRobot(ctx, r);
      }
    }
  }

  drawRobot(ctx, r) {
    const scr = this.worldToScreen(r.x, r.y);
    const color = this.robotColors[r.id] || {
      main: "#0284c7",
      glow: "rgba(2, 132, 199, 0.2)",
      dark: "#0369a1",
      bg: "#f0f9ff"
    };

    // --- Safety and LiDAR circles ---
    if (this.showLidar) {
      // 3.0m LiDAR range
      const rLidar = 3.0 * this.scale;
      ctx.beginPath();
      ctx.arc(scr.x, scr.y, rLidar, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(2, 132, 199, 0.2)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // 1.05m Slow margin
      const rSlow = 1.05 * this.scale;
      ctx.beginPath();
      ctx.arc(scr.x, scr.y, rSlow, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(217, 119, 6, 0.45)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // 0.50m Hard stop margin
      const rHard = 0.50 * this.scale;
      ctx.beginPath();
      ctx.arc(scr.x, scr.y, rHard, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(220, 38, 38, 0.55)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // --- AMR Physical Chassis Rendering ---
    ctx.save();
    ctx.translate(scr.x, scr.y);
    ctx.rotate(r.theta);

    const lengthPx = 1.0 * this.scale; // 1.0 m length
    const widthPx = 0.70 * this.scale;  // 0.70 m body width
    const wheelSepPx = 0.86 * this.scale; // 0.86 m wheel separation
    const wheelW = 0.12 * this.scale;
    const wheelL = 0.36 * this.scale;

    // Degraded / Yielding Halos
    if (r.degraded) {
      ctx.beginPath();
      ctx.arc(0, 0, lengthPx * 0.85, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(124, 58, 237, 0.2)";
      ctx.fill();
      ctx.strokeStyle = "#7c3aed";
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (r.yielding) {
      ctx.beginPath();
      ctx.arc(0, 0, lengthPx * 0.75, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(217, 119, 6, 0.2)";
      ctx.fill();
    }

    // Wheels (Left & Right - Dark Slate)
    ctx.fillStyle = "#1e293b";
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 1;

    // Left wheel
    ctx.fillRect(-wheelL / 2, -wheelSepPx / 2 - wheelW / 2, wheelL, wheelW);
    ctx.strokeRect(-wheelL / 2, -wheelSepPx / 2 - wheelW / 2, wheelL, wheelW);

    // Right wheel
    ctx.fillRect(-wheelL / 2, wheelSepPx / 2 - wheelW / 2, wheelL, wheelW);
    ctx.strokeRect(-wheelL / 2, wheelSepPx / 2 - wheelW / 2, wheelL, wheelW);

    // Main Chassis Body (Clean white body with distinct robot color border)
    ctx.beginPath();
    ctx.roundRect(-lengthPx / 2, -widthPx / 2, lengthPx, widthPx, 4);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.strokeStyle = color.main;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Directional Chevron Arrow (Robot's distinct color)
    ctx.fillStyle = color.main;
    ctx.beginPath();
    ctx.moveTo(lengthPx * 0.35, 0);
    ctx.lineTo(lengthPx * 0.15, -widthPx * 0.25);
    ctx.lineTo(lengthPx * 0.18, 0);
    ctx.lineTo(lengthPx * 0.15, widthPx * 0.25);
    ctx.closePath();
    ctx.fill();

    // Cargo Container if LOADED
    if (r.payload > 0 || r.mode === "LOADED" || r.mode === "TO_DROPOFF") {
      const boxSize = widthPx * 0.55;
      ctx.fillStyle = "#b45309"; // Warm cardboard
      ctx.fillRect(-boxSize / 2, -boxSize / 2, boxSize, boxSize);
      ctx.strokeStyle = "#78350f";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(-boxSize / 2, -boxSize / 2, boxSize, boxSize);

      // Package tape
      ctx.fillStyle = "#d97706";
      ctx.fillRect(-boxSize / 2, -2, boxSize, 4);
    }

    ctx.restore();

    // Robot ID & Status Callout (Always upright)
    ctx.font = "bold 11px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // Text pill badge with soft shadow
    const badgeW = 44;
    const badgeH = 17;
    const badgeY = scr.y - (lengthPx / 2 + 14);

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.roundRect(scr.x - badgeW / 2, badgeY - badgeH / 2, badgeW, badgeH, 4);
    ctx.fill();
    ctx.strokeStyle = color.main;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = color.main;
    ctx.fillText(`AMR ${r.id}`, scr.x, badgeY);

    // Speed readout under robot
    if (this.scale > 16) {
      ctx.fillStyle = "#475569";
      ctx.font = "bold 9px monospace";
      ctx.fillText(`${r.v.toFixed(2)}m/s`, scr.x, scr.y + lengthPx / 2 + 12);
    }
  }
}
