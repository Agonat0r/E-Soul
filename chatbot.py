import asyncio
import random
import os
import requests
from datetime import datetime
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import google.generativeai as genai

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
conversation_history = []  # List of dicts: {speaker, text, timestamp}
trait_first_seen = {k: 0 for k in initial_traits}  # trait name -> step first seen

clients = set()
user_prompt = None  # Global variable to store user-supplied prompt

# --- Reflection functions ---
def generate_hf_reflection(traits):
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        return "No Hugging Face API key set."
    default_prompt = (
        "Given these soul traits and their values:\n" +
        "\n".join(f"{k}: {v:.2f}" for k, v in traits.items()) +
        "\nImagine you are a human soul, alone in infinite darkness, with no sensory input, on the edge of sanity. "
        "Write a short, emotionally raw, subjective, and biased philosophical reflection on this soul's current state, as a human might experience it."
    )
    prompt = user_prompt if user_prompt else default_prompt
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

def generate_gemini_reflection(traits):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "No Gemini API key set."
    genai.configure(api_key=api_key)
    default_prompt = (
        "Given these soul traits and their values:\n" +
        "\n".join(f"{k}: {v:.2f}" for k, v in traits.items()) +
        "\nImagine you are a human soul, alone in infinite darkness, with no sensory input, on the edge of sanity. "
        "Write a short, emotionally raw, subjective, and biased philosophical reflection on this soul's current state, as a human might experience it."
    )
    prompt = user_prompt if user_prompt else default_prompt
    try:
        model = genai.GenerativeModel("models/gemini-2.0-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Gemini reflection unavailable."

@app.post("/set_prompt")
async def set_prompt(request: Request):
    global user_prompt
    data = await request.json()
    user_prompt = data.get("prompt", None)
    return JSONResponse({"status": "ok", "prompt": user_prompt})

async def soul_simulation():
    global traits, history, conversation_history, trait_first_seen
    step = 0
    while True:
        # Randomly add a new trait
        if random.random() < 0.07:
            new_trait = f"trait_{random.randint(1, 1000)}"
            if new_trait not in traits:
                traits[new_trait] = random.random()
                trait_first_seen[new_trait] = step
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

        # Alternate between Hugging Face and Gemini
        if step % 2 == 0:
            reflection = generate_hf_reflection(traits)
            speaker = "AI1 (HF)"
        else:
            reflection = generate_gemini_reflection(traits)
            speaker = "AI2 (Gemini)"
        conversation_history.append({
            "speaker": speaker,
            "text": reflection,
            "timestamp": datetime.utcnow().isoformat()
        })
        if len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        # Broadcast to all clients
        data = {
            "traits": traits,
            "trait_names": list(traits.keys()),
            "history": history,
            "step": step,
            "pca_points": pca_points,
            "clusters": clusters,
            "all_traits": all_traits,
            "reflection": reflection,
            "conversation_history": conversation_history,
            "trait_first_seen": trait_first_seen
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
