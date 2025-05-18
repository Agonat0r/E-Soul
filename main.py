import asyncio
import random
import os
import requests
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

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

def generate_reflection(traits):
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        return "No Hugging Face API key set."
    prompt = (
        "Given these soul traits and their values:\n" +
        "\n".join(f"{k}: {v:.2f}" for k, v in traits.items()) +
        "\nWrite a short, unbiased, philosophical reflection on this soul's current state."
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 80}
    }
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers=headers, json=payload, timeout=10
        )
        if response.status_code == 200:
            return response.json()[0]["generated_text"].strip()
        else:
            return "Reflection unavailable."
    except Exception:
        return "Reflection unavailable."

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
        # Evolve all traits (random walk, no target)
        for k in traits:
            traits[k] = min(1.0, max(0.0, traits[k] + random.uniform(-0.05, 0.05)))
        # Save snapshot (deep copy)
        history.append(traits.copy())
        if len(history) > 200:
            history = history[-200:]

        # --- PCA and KMeans ---
        all_traits = sorted({k for snap in history for k in snap})
        X = np.array([[snap.get(k, 0) for k in all_traits] for snap in history])
        pca_points = []
        clusters = []
        if len(history) > 2:
            pca = PCA(n_components=2)
            pca_points = pca.fit_transform(X).tolist()
            kmeans = KMeans(n_clusters=min(4, len(history)), n_init=10)
            clusters = kmeans.fit_predict(X).tolist()

        reflection = generate_reflection(traits)

        # Broadcast to all clients
        data = {
            "traits": traits,
            "trait_names": list(traits.keys()),
            "history": history,
            "step": step,
            "pca_points": pca_points,
            "clusters": clusters,
            "all_traits": all_traits,
            "reflection": reflection
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