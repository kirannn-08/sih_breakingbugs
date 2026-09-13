const canvas = document.getElementById('sim');
const ctx = canvas.getContext('2d');
const leftPanel = document.getElementById('left-panel');

let width, height;
function resize() {
    width = leftPanel.clientWidth;
    height = leftPanel.clientHeight;
    canvas.width = width;
    canvas.height = height;
}
window.addEventListener('resize', resize);
resize();

// -----------------------------------------------------
// STATE & DATA
// -----------------------------------------------------
let amrs = {}; // {id: {x, y, in_dz, is_bridge, packets_sent: 0}}
let serverNode = {x: 500, y: 40, w: 160, h: 50};
let topologyLinks = []; 
let activeP2PLines = []; // {src, dst, type, timestamp}
let packetsInFlight = []; // {src, dst, type, timestamp}
let deadzone = {x: 300, y: 200, w: 400, h: 300, isDragging: false, offsetX: 0, offsetY: 0};

const amrColors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6', '#34495e', '#1abc9c', '#f1c40f', '#e84393', '#00cec9'];

// Ensure server is registered
amrs[0] = {id: 0, x: serverNode.x, y: serverNode.y, in_dz: false, is_bridge: false};

// -----------------------------------------------------
// WEBSOCKET CONNECTION
// -----------------------------------------------------
const ws = new WebSocket(`ws://${location.host}/ws`);

ws.onopen = () => {
    logMessage('log-server', 'SYS', 'Backend', 'WebSocket Connected to Python Backend', 'server-broadcast');
};

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    const type = msg.type;
    const data = msg.data;
    const pkt = msg.packet;

    if (type === "SYNC_FRAME") {
        data.amrs.forEach(amr_data => {
            const id = amr_data.id;
            if (!amrs[id]) amrs[id] = { id, in_dz: false, is_bridge: false };
            amrs[id].x = amr_data.x;
            amrs[id].y = amr_data.y;
        });
        topologyLinks = data.links;
        // Reset bridge status
        Object.values(amrs).forEach(a => a.is_bridge = false);
    }
    else if (type === "WIFI_LOST") {
        if(amrs[data.amr_id]) {
            amrs[data.amr_id].in_dz = true;
            amrs[data.amr_id].is_bridge = false;
        }
    }
    else if (type === "WIFI_RECOVERED") {
        if(amrs[data.amr_id]) amrs[data.amr_id].in_dz = false;
    }
    else if (type === "BRIDGE_SELECTED") {
        if(amrs[data.amr_id]) amrs[data.amr_id].is_bridge = true;
    }
    else if (type === "PACKET_TRANSMITTING") {
        const src = amrs[pkt.forwarder_id];
        let dst = amrs[pkt.destination_id];
        
        if (src) {
            
            if (pkt.destination_id != -1 && dst) {
                // Point to point
                packetsInFlight.push({src: src, dst: dst, type: pkt.medium, timestamp: Date.now()});
                
                if (pkt.origin_id !== 0 && pkt.destination_id !== 0) {
                     activeP2PLines.push({src: src, dst: dst, type: pkt.medium, timestamp: Date.now()});
                }
                
                // Log it if it's task related
                if (pkt.message_type === "TASK" || pkt.message_type === "ACK") {
                    let logType = pkt.medium === 'WIFI' ? 'wifi-wifi' : 'wisun-wisun';
                    if (pkt.forwarder_id === 0 || pkt.destination_id === 0) {
                        logType = 'server-broadcast';
                    } else if (amrs[pkt.forwarder_id].is_bridge || dst.is_bridge) {
                        logType = 'relay';
                    }
                    
                    let p_id = pkt.destination_id === 0 ? 'server' : `amr-${pkt.destination_id}`;
                    logMessage(`log-${p_id}`, pkt.forwarder_id===0?'Server':`AMR ${pkt.forwarder_id}`, pkt.destination_id===0?'Server':`AMR ${pkt.destination_id}`, `[${pkt.message_type}] ID:${pkt.message_id}`, logType);
                }
            } else {
                // Broadcast
                topologyLinks.forEach(l => {
                    if (l.src === pkt.forwarder_id && amrs[l.dst]) {
                        packetsInFlight.push({src: src, dst: amrs[l.dst], type: l.type, timestamp: Date.now()});
                        if (pkt.origin_id !== 0 && l.dst !== 0) {
                             activeP2PLines.push({src: src, dst: amrs[l.dst], type: l.type, timestamp: Date.now()});
                        }
                    }
                    if (l.dst === pkt.forwarder_id && amrs[l.src]) {
                        packetsInFlight.push({src: src, dst: amrs[l.src], type: l.type, timestamp: Date.now()});
                        if (pkt.origin_id !== 0 && l.src !== 0) {
                             activeP2PLines.push({src: src, dst: amrs[l.src], type: l.type, timestamp: Date.now()});
                        }
                    }
                });
                
                // Log Telemetry Broadcast
                if (pkt.message_type === "TELEMETRY" && pkt.origin_id !== 0) {
                    let logType = pkt.medium === 'WIFI' ? 'wifi-wifi' : 'wisun-wisun';
                    if (amrs[pkt.forwarder_id].is_bridge) logType = 'relay';
                    
                    const nowTs = Date.now();
                    if (!src.lastTelLog || nowTs - src.lastTelLog > 1500) {
                        src.lastTelLog = nowTs;
                        const p = pkt.payload;
                        if (p) {
                            const px = Math.round(p.position?.x || 0);
                            const py = Math.round(p.position?.y || 0);
                            const v = (p.velocity?.linear || 0).toFixed(1);
                            const stat = src.in_dz ? 'Wi-SUN Mesh' : 'Wi-Fi AP';
                            
                            let msg = `Pos:(${px},${py}) | Vel:${v} | Task:${p.intent || 'Idle'} | Net:${stat}`;
                            logMessage(`log-amr-${pkt.forwarder_id}`, `AMR ${pkt.forwarder_id}`, 'ALL', msg, logType);
                        }
                    }
                }
            }
        }
    }
    else if (type === "TASK_RECEIVED") {
        logMessage(`log-amr-${data.amr_id}`, 'System', `AMR ${data.amr_id}`, `Received Task Assignment`, 'server-broadcast');
    }
    else if (type === "ACK_RECEIVED") {
        if (data.amr_id === 0) { // Server received ACK
            let allotmentMsg = `[ASSIGNMENT] Task allotted to AMR ${pkt.origin_id}`;
            logMessage('log-server', `AMR ${pkt.origin_id}`, 'Server', allotmentMsg, 'server-broadcast');
        } else {
            logMessage(`log-amr-${data.amr_id}`, `AMR ${pkt.origin_id}`, `AMR ${data.amr_id}`, `Ack received`, 'wifi-wifi');
        }
    }
};

// -----------------------------------------------------
// UI LOGGING
// -----------------------------------------------------
function logMessage(panelId, sender, receiver, msg, type) {
    const panel = document.getElementById(panelId);
    if(!panel) return;
    const div = document.createElement('div');
    div.className = `msg ${type}`;
    const time = new Date().toLocaleTimeString();
    div.innerHTML = `[${time}] <b>${sender} &rarr; ${receiver}:</b> ${msg}`;
    panel.appendChild(div);
    
    if (panel.children.length > 40) {
        panel.removeChild(panel.firstChild);
    }
    
    panel.scrollTop = panel.scrollHeight;
}

window.generateTask = function() {
    let taskName = document.getElementById('task-name').value || 'Transport Pallet';
    let dest_id = -1; // Broadcast to all initially
    ws.send(JSON.stringify({action: "SEND_TASK", src: 0, dst: dest_id, task: {type: "GOTO_BAY", name: taskName}}));
    logMessage('log-server', 'Server', 'ALL', `Auctioning Task: ${taskName}`, 'server-broadcast');
};

// -----------------------------------------------------
// INTERACTIVITY (DEADZONE)
// -----------------------------------------------------
leftPanel.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    let mx = e.clientX, my = e.clientY;
    if (mx > deadzone.x && mx < deadzone.x + deadzone.w &&
        my > deadzone.y && my < deadzone.y + deadzone.h) {
        deadzone.isDragging = true;
        deadzone.offsetX = mx - deadzone.x;
        deadzone.offsetY = my - deadzone.y;
    }
});
window.addEventListener('mousemove', e => {
    if (deadzone.isDragging) {
        deadzone.x = e.clientX - deadzone.offsetX;
        deadzone.y = e.clientY - deadzone.offsetY;
        ws.send(JSON.stringify({action: "UPDATE_DZ", x0: deadzone.x, y0: deadzone.y, x1: deadzone.x+deadzone.w, y1: deadzone.y+deadzone.h}));
    }
});
window.addEventListener('mouseup', () => {
    deadzone.isDragging = false;
});


// -----------------------------------------------------
// RENDERING
// -----------------------------------------------------
function drawMap() {
    ctx.strokeStyle = 'rgba(189, 195, 199, 0.3)';
    ctx.lineWidth = 1;
    for (let i = 0; i < width; i += 40) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke();
    }
    for (let i = 0; i < height; i += 40) {
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke();
    }
    
    ctx.fillStyle = '#2c3e50';
    ctx.beginPath();
    ctx.roundRect(serverNode.x - serverNode.w/2, serverNode.y - serverNode.h/2, serverNode.w, serverNode.h, 10);
    ctx.fill();
    ctx.fillStyle = '#ecf0f1';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('CENTRAL LIVE TASK SERVER', serverNode.x, serverNode.y + 6);
    
    let startX = 120;
    let colWidth = 160;
    for (let i = 0; i < 6; i++) {
        let x = startX + i * colWidth;
        let y = 130;
        let grad = ctx.createRadialGradient(x, y, 10, x, y, 25);
        grad.addColorStop(0, 'rgba(230, 126, 34, 0.5)');
        grad.addColorStop(1, 'rgba(230, 126, 34, 0)');
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(x, y, 25, 0, Math.PI*2); ctx.fill();
        
        ctx.fillStyle = '#d35400';
        ctx.beginPath(); ctx.arc(x, y, 16, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = 'bold 12px Arial'; ctx.fillText('\u26A1', x, y+4);
        ctx.fillStyle = '#333'; ctx.fillText(`C${i+1}`, x, y + 40);
    }
    
    ctx.fillStyle = '#ecf0f1';
    ctx.strokeStyle = '#bdc3c7';
    ctx.lineWidth = 2;
    
    const drawRack = (x, y) => {
        const w = 70; const h = 120;
        ctx.beginPath(); ctx.rect(x - w/2, y - h/2, w, h); ctx.fill(); ctx.stroke();
        ctx.beginPath();
        for(let i=1; i<3; i++) { ctx.moveTo(x - w/2 + i*23.3, y - h/2); ctx.lineTo(x - w/2 + i*23.3, y + h/2); }
        for(let i=1; i<5; i++) { ctx.moveTo(x - w/2, y - h/2 + i*24); ctx.lineTo(x + w/2, y - h/2 + i*24); }
        ctx.stroke();
        
        ctx.strokeStyle = '#3498db'; ctx.setLineDash([4, 4]);
        ctx.strokeRect(x - w/2 - 18, y - 10, 18, 20); ctx.strokeRect(x + w/2, y - 10, 18, 20);
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#3498db'; ctx.font = '9px Arial';
        ctx.fillText('BAY', x - w/2 - 9, y + 3); ctx.fillText('BAY', x + w/2 + 9, y + 3);
        ctx.strokeStyle = '#bdc3c7'; 
    };
    
    for (let row = 0; row < 2; row++) {
        for (let col = 0; col < 6; col++) {
            if (col === 2 || col === 5) continue; 
            drawRack(startX + col * colWidth, 300 + row * 220);
        }
    }
}

function drawDeadzone() {
    ctx.fillStyle = 'rgba(231, 76, 60, 0.15)';
    ctx.fillRect(deadzone.x, deadzone.y, deadzone.w, deadzone.h);
    
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 10]);
    ctx.strokeRect(deadzone.x, deadzone.y, deadzone.w, deadzone.h);
    ctx.setLineDash([]);
    
    ctx.fillStyle = '#c0392b';
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('\u26A0\uFE0F WI-FI DEADZONE', deadzone.x + deadzone.w/2, deadzone.y + 35);
}

function loop() {
    ctx.clearRect(0, 0, width, height);
    drawMap();
    drawDeadzone();
    
    const now = Date.now();
    
    // Draw Topology Links (Hide messy server links)
    topologyLinks.forEach(link => {
        
        const src = amrs[link.src];
        const dst = amrs[link.dst];
        if (src && dst) {
            ctx.beginPath();
            ctx.moveTo(src.x, src.y);
            ctx.lineTo(dst.x, dst.y);
            
            if (src.in_dz && dst.in_dz) {
                // Both in deadzone
                ctx.strokeStyle = 'rgba(230, 126, 34, 0.15)';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);
            } else if (!src.in_dz && !dst.in_dz) {
                // Both in Wi-Fi
                ctx.strokeStyle = 'rgba(52, 152, 219, 0.15)';
                ctx.lineWidth = 1;
                ctx.setLineDash([]);
            } else {
                // Bridge
                ctx.strokeStyle = 'rgba(155, 89, 182, 0.15)';
                ctx.lineWidth = 1;
                ctx.setLineDash([]);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }
    });
    
    // Draw Active P2P Communication Lines
    activeP2PLines = activeP2PLines.filter(l => now - l.timestamp < 300);
    activeP2PLines.forEach(l => {
        ctx.beginPath();
        ctx.moveTo(l.src.x, l.src.y);
        ctx.lineTo(l.dst.x, l.dst.y);
        
        const alpha = 1.0 - ((now - l.timestamp) / 300);
        ctx.lineWidth = 4;
        
        if (l.src.in_dz && l.dst.in_dz) {
            // Wi-SUN to Wi-SUN (Orange Dashed)
            ctx.strokeStyle = `rgba(230, 126, 34, ${alpha})`;
            ctx.setLineDash([10, 5]);
        } else if (!l.src.in_dz && !l.dst.in_dz) {
            // Wi-Fi to Wi-Fi (Red Solid)
            ctx.strokeStyle = `rgba(231, 76, 60, ${alpha})`;
            ctx.setLineDash([]);
        } else {
            // Bridge: Deadzone <-> Wi-Fi (Blue Solid)
            ctx.strokeStyle = `rgba(52, 152, 219, ${alpha})`;
            ctx.setLineDash([]);
        }
        
        ctx.stroke();
        ctx.setLineDash([]);
    });

    // Draw Packets (Animated)
    packetsInFlight = packetsInFlight.filter(p => now - p.timestamp < 300);
    packetsInFlight.forEach(pkt => {
        const progress = (now - pkt.timestamp) / 300;
        const x = pkt.src.x + (pkt.dst.x - pkt.src.x) * progress;
        const y = pkt.src.y + (pkt.dst.y - pkt.src.y) * progress;
        
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI*2);
        ctx.fillStyle = pkt.type === 'WIFI' ? '#3498db' : '#d35400';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    });

    // Draw AMRs
    for (const [idStr, amr] of Object.entries(amrs)) {
        if (idStr == 0) continue; // Server is static, drawn in map
        
        ctx.save();
        ctx.translate(amr.x, amr.y);
        
        // Body
        ctx.fillStyle = amrColors[(amr.id - 1) % amrColors.length];
        ctx.beginPath();
        ctx.roundRect(-15, -20, 30, 40, 5);
        ctx.fill();
        
        // Status light
        ctx.beginPath();
        ctx.arc(20, -25, 6, 0, Math.PI*2);
        ctx.fillStyle = amr.in_dz ? '#e74c3c' : '#3498db';
        if(amr.in_dz && amr.is_bridge) ctx.fillStyle = '#f1c40f'; // Bridge
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        ctx.restore();
        
        ctx.fillStyle = '#333';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`AMR ${amr.id}`, amr.x, amr.y - 25);
        
        if (amr.is_bridge) {
            ctx.fillStyle = '#f1c40f';
            ctx.font = 'bold 10px Arial';
            ctx.fillText('BRIDGE', amr.x, amr.y + 30);
        }
    }
    
    requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
