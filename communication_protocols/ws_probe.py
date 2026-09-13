import asyncio
import websockets
import json

async def probe():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send(json.dumps({"action": "SEND_TASK", "src": 0, "dst": -1, "task": {"type": "GOTO_BAY"}}))
        count = 0
        while count < 10:
            msg = await ws.recv()
            data = json.loads(msg)
            if data["type"] in ["PACKET_TRANSMITTING", "BRIDGE_SELECTED", "TASK_RECEIVED"]:
                if data.get("packet", {}).get("message_type") != "TELEMETRY":
                    print(data["type"], data["packet"]["message_type"], data["packet"]["forwarder_id"], data["packet"]["destination_id"], data["packet"]["medium"])
                    count += 1

if __name__ == "__main__":
    asyncio.run(probe())
