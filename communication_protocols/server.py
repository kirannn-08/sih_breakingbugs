import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from simulation import SimulationController
import dataclasses

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

manager = ConnectionManager()

def on_sim_event(event_type, packet=None, **kwargs):
    allowed_events = {"SYNC_FRAME", "WIFI_LOST", "WIFI_RECOVERED", "BRIDGE_SELECTED", "PACKET_TRANSMITTING", "TASK_RECEIVED", "ACK_RECEIVED"}
    if event_type not in allowed_events:
        return

    event_data = {
        "type": event_type,
        "data": kwargs
    }
    if packet:
        # Convert dataclass to dict
        pkt_dict = dataclasses.asdict(packet)
        event_data["packet"] = pkt_dict
        
    # Use event loop to broadcast safely
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(manager.broadcast(event_data))
    except RuntimeError:
        pass

sim = SimulationController(num_amrs=6, on_event=on_sim_event)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(sim.run_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            
            if cmd.get("action") == "FORCE_DEADZONE":
                amr_id = cmd.get("amr_id")
                in_dz = cmd.get("in_deadzone")
                sim.force_dead_zone(amr_id, in_dz)
                
            elif cmd.get("action") == "UPDATE_DZ":
                x0 = cmd.get("x0")
                y0 = cmd.get("y0")
                x1 = cmd.get("x1")
                y1 = cmd.get("y1")
                sim.set_dead_zone(x0, y0, x1, y1)
                
            elif cmd.get("action") == "SEND_TASK":
                src = cmd.get("src")
                dst = cmd.get("dst")
                task = cmd.get("task")
                sim.create_task(src, dst, task)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
