import asyncio
import random
import os
import requests
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity # Added for trait similarity
import numpy as np
import google.generativeai as genai
import google.api_core.exceptions # To specifically catch NotFound
import traceback

# Your FastAPI application instance
app = FastAPI()

# CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Trait Similarity Colors ---
DEFAULT_COLORS_PALETTE = [
    "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#6A5ACD", 
    "#FF69B4", "#00CED1", "#FFA07A", "#9370DB", "#3CB371",
    "#F08080", "#ADD8E6", "#90EE90", "#FFFFE0", "#C8A2C8",
    "#DB7093", "#AFEEEE", "#F5DEB3", "#DDA0DD", "#8FBC8F",
    "#FA8072", "#B0C4DE", "#98FB98", "#FAFAD2", "#E6E6FA" 
]
DEFAULT_TRAIT_COLOR = "#B0B0B0" # Default grey for traits if no other color assigned

# === Dynamic Soul Trait Logic ===
initial_traits = {"empathy": 0.5, "curiosity": 0.5, "self_awareness": 0.5}
traits = initial_traits.copy()
history = []
conversation_history = []
trait_first_seen = {k: 0 for k in initial_traits}

clients = set()
user_prompt = None
user_hf_api_key = None
user_gemini_api_key = None

# --- Trait Similarity Calculation Function ---
def calculate_trait_similarities_and_colors(
    trait_names_with_history: list[str], 
    trait_history_matrix: np.ndarray, 
    similarity_threshold: float = 0.95,
    min_history_steps_for_similarity: int = 3,
    colors_palette: list[str] = None
) -> dict[str, str]:
    if colors_palette is None:
        colors_palette = DEFAULT_COLORS_PALETTE
    initial_colors_for_historical_traits: dict[str, str] = {
        name: colors_palette[i % len(colors_palette)] 
        for i, name in enumerate(trait_names_with_history)
    }
    if (not trait_names_with_history or 
        len(trait_names_with_history) < 2 or 
        not isinstance(trait_history_matrix, np.ndarray) or
        trait_history_matrix.ndim != 2 or 
        trait_history_matrix.shape[1] != len(trait_names_with_history) or 
        trait_history_matrix.shape[0] < min_history_steps_for_similarity):
        return initial_colors_for_historical_traits
    num_historical_traits = len(trait_names_with_history)
    try:
        similarity_matrix = cosine_similarity(trait_history_matrix.T)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}. Falling back to unique colors.")
        traceback.print_exc()
        return initial_colors_for_historical_traits
    assigned_mask = [False] * num_historical_traits
    current_color_index = 0
    grouped_colors: dict[str, str] = {}
    for i in range(num_historical_traits):
        if assigned_mask[i]:
            continue
        current_color = colors_palette[current_color_index % len(colors_palette)]
        grouped_colors[trait_names_with_history[i]] = current_color
        assigned_mask[i] = True
        for j in range(i + 1, num_historical_traits):
            if not assigned_mask[j] and similarity_matrix[i, j] >= similarity_threshold:
                grouped_colors[trait_names_with_history[j]] = current_color
                assigned_mask[j] = True
        current_color_index += 1
    final_colors_for_historical_traits = initial_colors_for_historical_traits.copy()
    final_colors_for_historical_traits.update(grouped_colors)
    return final_colors_for_historical_traits

# --- Reflection Generation Functions ---
def generate_hf_reflection(traits):
    api_key = user_hf_api_key or os.getenv("HF_API_KEY")
    if not api_key:
        return "No Hugging Face API key set."
    default_prompt = (
        "Given these soul traits and their values:\n" +
        "\n".join(f"{k}: {v:.2f}" for k, v in traits.items()) +
        "\nImagine you are a human soul, alone in infinite darkness, with no sensory input, on the edge of sanity. "
        "You have a lifespan of potentially 1000 steps. "
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
    api_key = user_gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "No Gemini API key set."
    genai.configure(api_key=api_key)
    default_prompt = (
        "Given these soul traits and their values:\n" +
        "\n".join(f"{k}: {v:.2f}" for k, v in traits.items()) +
        "\nImagine you are a human soul, alone in infinite darkness, with no sensory input, on the edge of sanity. "
        "You have a lifespan of potentially 1000 steps. "
        "Write a short, emotionally raw, subjective, and biased philosophical reflection on this soul's current state, as a human might experience it."
    )
    prompt = user_prompt if user_prompt else default_prompt
    try:
        model = genai.GenerativeModel("models/gemini-2.0-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Gemini reflection unavailable."

# --- API Endpoints ---
@app.post("/set_prompt")
async def set_prompt_endpoint(request: Request):
    global user_prompt; data = await request.json(); new_prompt = data.get("prompt")
    if isinstance(new_prompt, str) and new_prompt.strip(): user_prompt = new_prompt
    elif new_prompt is None or (isinstance(new_prompt, str) and not new_prompt.strip()): user_prompt = None
    else: return JSONResponse({"status":"error","message":"Invalid prompt format"},status_code=400)
    return JSONResponse({"status":"ok","prompt_set_to":user_prompt or "default"})

@app.post("/set_api_keys")
async def set_api_keys(request: Request):
    global user_hf_api_key, user_gemini_api_key
    data = await request.json()
    hf_key = data.get("hf_api_key")
    gemini_key = data.get("gemini_api_key")
    errors = {}
    # Validate Hugging Face key
    if hf_key:
        headers = {"Authorization": f"Bearer {hf_key}"}
        payload = {"inputs": "Test"}
        try:
            resp = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                headers=headers, json=payload, timeout=10
            )
            if resp.status_code == 401:
                errors["hf_api_key"] = "Invalid Hugging Face API key."
            elif resp.status_code != 200:
                errors["hf_api_key"] = f"Hugging Face error: {resp.status_code}"
            else:
                user_hf_api_key = hf_key
        except Exception as e:
            errors["hf_api_key"] = f"Hugging Face error: {str(e)}"
    else:
        errors["hf_api_key"] = "No Hugging Face API key provided."
    # Validate Gemini key
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("models/gemini-2.0-pro")
            response = model.generate_content("Test")
            if not hasattr(response, 'text') or not response.text:
                errors["gemini_api_key"] = "Invalid Gemini API key."
            else:
                user_gemini_api_key = gemini_key
        except Exception as e:
            errors["gemini_api_key"] = f"Gemini error: {str(e)}"
    else:
        errors["gemini_api_key"] = "No Gemini API key provided."
    if errors:
        return JSONResponse({"status": "error", "errors": errors}, status_code=400)
    return JSONResponse({"status": "ok"})

# --- Background Simulation Task ---
async def soul_simulation():
    global traits, history, conversation_history, trait_first_seen
    step = 0
    while True:
        # Randomize probabilities for this step
        add_prob = random.uniform(0.01, 0.15)  # Random chance for adding a trait
        remove_prob = random.uniform(0.01, 0.10)  # Random chance for removing a trait
        # Randomly add a new trait
        if random.random() < add_prob:
            new_trait = f"trait_{random.randint(1, 1000)}"
            if new_trait not in traits:
                traits[new_trait] = random.random()
                trait_first_seen[new_trait] = step
        # Randomly remove a trait (but keep at least 2)
        if len(traits) > 2 and random.random() < remove_prob:
            to_remove = random.choice(list(traits.keys()))
            del traits[to_remove]
        # Evolve all traits (random walk, no target)
        for k in traits:
            traits[k] = min(1.0, max(0.0, traits[k] + random.uniform(-0.05, 0.05)))
        # Save snapshot (deep copy)
        history.append(traits.copy())
        if len(history) > 200:
            history = history[-200:]
        aptl, pcp, cl, X = [], [], [], np.array([])
        mhfa = 3
        if len(history) >= mhfa:
            ats = {k for s in history for k in s}
            aptl = sorted(list(ats))
            if aptl:
                Xl = [[s.get(kt, 0.0) for kt in aptl] for s in history]
                X = np.array(Xl)
                if X.shape[0] >= mhfa and X.shape[1] > 0:
                    npc = min(2, X.shape[0], X.shape[1])
                    if npc >= 1:
                        try:
                            pca = PCA(n_components=npc)
                            pcp = pca.fit_transform(X).tolist()
                        except Exception as ep:
                            print(f"S{step} PCA err:{ep}")
                    nkc = min(4, X.shape[0])
                    if nkc >= 1:
                        try:
                            km = KMeans(n_clusters=nkc, n_init='auto', random_state=0)
                            cl = km.fit_predict(X).tolist()
                        except Exception as ek:
                            print(f"S{step} KM err:{ek}")
        
        ctna = list(traits.keys())
        ftc = {}
        cfs = {}
        if aptl and X.ndim == 2 and X.shape[0] >= 1 and X.shape[1] == len(aptl):
            cfs = calculate_trait_similarities_and_colors(trait_names_with_history=aptl, trait_history_matrix=X, min_history_steps_for_similarity=mhfa)
        ftc.update(cfs)
        uc = set(ftc.values())
        ap = [c for c in DEFAULT_COLORS_PALETTE if c not in uc]
        if not ap:
            ap = DEFAULT_COLORS_PALETTE
        cin = 0
        for n in ctna:
            if n not in ftc:
                if ap:
                    ftc[n] = ap[cin % len(ap)]
                    cin += 1
                else:
                    ftc[n] = DEFAULT_TRAIT_COLOR
        ftc = {n: ftc.get(n, DEFAULT_TRAIT_COLOR) for n in ctna}

        rt = "Ref pend..."
        sn = "Sys"
        ctfr = traits.copy()
        if step % 2 == 0:
            rt = generate_hf_reflection(ctfr)
            sn = "AI1(HF)"
        else:
            rt = generate_gemini_reflection(ctfr)
            sn = "AI2(Gemini)"
        conversation_history.append({"speaker": sn, "text": rt, "timestamp": datetime.now(timezone.utc).isoformat()})
        if len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]
        
        data_to_send = {"traits": traits, "trait_names": ctna, "history": history, "step": step, "pca_points": pcp, "clusters": cl,
                      "all_traits_pca_order": aptl, "reflection": rt, "conversation_history": conversation_history,
                      "trait_first_seen": trait_first_seen, "trait_colors": ftc}
        for wc in list(clients):
            try:
                await wc.send_json(data_to_send)
            except:
                clients.discard(wc)
        step += 1
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(soul_simulation())
    print("Soul sim task created.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    ch = websocket.client.host or "Unk"
    cp = websocket.client.port or "N/A"
    print(f"Client {ch}:{cp} connected. Total clients:{len(clients)}")
    try:
        while True:
            await asyncio.sleep(60)
    except:
        pass
    finally:
        clients.discard(websocket)
        print(f"Client {ch}:{cp} disconnected. Total clients:{len(clients)}")

static_dir_name = "static"
if not os.path.exists(static_dir_name):
    os.makedirs(static_dir_name)
    dummy_html_path = os.path.join(static_dir_name, "index.html")
    if not os.path.exists(dummy_html_path):
        with open(dummy_html_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Soul Sim</title><style>body{font-family:sans-serif;margin:20px} pre{background-color:#f4f4f4;padding:10px;border-radius:5px;white-space:pre-wrap;word-wrap:break-word;max-height:600px;overflow-y:auto}</style></head>
<body><h1>Soul Sim Monitor 🚀</h1><div id="traits-container"></div><pre id="data_display">Connecting...</pre>
<script>
const ws=new WebSocket(`${window.location.protocol==='https:'?'wss:':'ws:'}//${location.host}/ws`);
const dd=document.getElementById('data_display'),tc=document.getElementById('traits-container');
ws.onopen=()=>{dd.textContent='Connected. Waiting for data...';};
ws.onmessage=(evt)=>{const d=JSON.parse(evt.data);console.log('Data:',d);dd.textContent=JSON.stringify(d,null,2);
tc.innerHTML='<h2>Current Traits:</h2>';if(d.traits&&d.trait_colors){const ul=document.createElement('ul');for(const tn in d.traits){const li=document.createElement('li');li.textContent=`${tn}: ${d.traits[tn].toFixed(2)}`;if(d.trait_colors[tn]){li.style.color=d.trait_colors[tn];li.style.fontWeight='bold';}ul.appendChild(li);}tc.appendChild(ul);}};
ws.onclose=(evt)=>{dd.textContent=`Disconnected. Code:${evt.code},Reason:${evt.reason||'N/A'}. Reloading...`;setTimeout(()=>window.location.reload(),5000);};
ws.onerror=(err)=>{console.error('WS Error:',err);dd.textContent='WS error.';};
</script></body></html>""")

app.mount("/", StaticFiles(directory=static_dir_name, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    mn = "chatbot"
    p = int(os.getenv("PORT", 10000))
    print(f"🚀 Starting Uvicorn server on http://0.0.0.0:{p}")
    print(f"👉 Running FastAPI 'app' from '{mn}.py'")
    print("🔑 Ensure API keys are set in environment.")
    uvicorn.run(f"{mn}:app", host="0.0.0.0", port=p, reload=True)