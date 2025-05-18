import asyncio
import random
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DYNAMIC SOUL TRAIT LOGIC ===
initial_traits = {"empathy": 0.5, "curiosity": 0.5, "self_awareness": 0.5}
traits = initial_traits.copy()
history = []  # List of dicts: each is a snapshot of traits at a step

clients = set()

async def soul_simulation():
    global traits, history
    step = 0
    while True:
        # Randomly add a new trait
        if random.random() < 0.07:
            new_trait = f"trait_{random.randint(1, 1000)}"
            if new_trait not in traits:
                traits[new_trait] = random.random()
        # Randomly remove a trait (but keep at least 2)
        if len(traits) > 2 and random.random() < 0.05:
            to_remove = random.choice(list(traits.keys()))
            del traits[to_remove]
        # Evolve all traits
        for k in traits:
            traits[k] = min(1.0, max(0.0, traits[k] + random.uniform(-0.05, 0.05)))
        # Save snapshot (deep copy)
        history.append(traits.copy())
        # Limit history length for memory
        if len(history) > 200:
            history = history[-200:]
        # Broadcast to all clients
        data = {
            "traits": traits,
            "trait_names": list(traits.keys()),
            "history": history,
            "step": step
        }
        for ws in list(clients):
            try:
                await ws.send_json(data)
            except Exception:
                clients.remove(ws)
        step += 1
        await asyncio.sleep(2)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(soul_simulation())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception:
        clients.remove(websocket)

# Serve static files (frontend)
app.mount("/", StaticFiles(directory="static", html=True), name="static") 