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
import numpy as np
import google.generativeai as genai
import traceback # For detailed error logging

# Your FastAPI application instance
app = FastAPI()

# CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for simplicity; restrict in production if needed
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods
    allow_headers=["*"], # Allows all headers
)

# === Dynamic Soul Trait Logic ===
initial_traits = {"empathy": 0.5, "curiosity": 0.5, "self_awareness": 0.5}
traits = initial_traits.copy() # Current state of traits
history = []  # List of dictionaries, each a snapshot of traits at a step
conversation_history = []  # List of dictionaries: {speaker, text, timestamp}
trait_first_seen = {k: 0 for k in initial_traits}  # Trait name -> step first seen

clients = set() # Set of connected WebSocket clients
user_prompt = None  # Global variable for user-supplied reflection prompt

# --- Reflection Generation Functions ---

def generate_hf_reflection(traits_input: dict) -> str:
    """Generates a reflection using a Hugging Face model."""
    api_key = os.getenv("HF_API_KEY")
    print(f"generate_hf_reflection: HF_API_KEY is {'SET' if api_key else 'NOT SET'}.")
    if not api_key:
        return "No Hugging Face API key set. Please check environment variables."
    
    # Default prompt structure
    current_prompt_text = (
        "Given these soul traits and their values:\n" +
        "\n".join(f"{k}: {v:.2f}" for k, v in traits_input.items()) +
        "\nImagine you are a human soul, alone in infinite darkness, with no sensory input, on the edge of sanity. "
        "Write a short, emotionally raw, subjective, and biased philosophical reflection on this soul's current state, as a human might experience it."
    )
    
    prompt_to_use = user_prompt if user_prompt else current_prompt_text
    
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": prompt_to_use,
        "parameters": {"max_new_tokens": 150, "temperature": 0.75, "return_full_text": False}, # Adjusted parameters
        "options": {"wait_for_model": True, "use_cache": False} # Options for inference API
    }
    
    hf_model_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    print(f"generate_hf_reflection: Sending request to {hf_model_url} with prompt starting: '{prompt_to_use[:70]}...'")

    try:
        response = requests.post(hf_model_url, headers=headers, json=payload, timeout=35) # Increased timeout
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result[0]["generated_text"].strip()
            print("generate_hf_reflection: Successfully received response from Hugging Face.")
            return generated_text
        else:
            print(f"generate_hf_reflection: Hugging Face API error. Status: {response.status_code}, Response: {response.text}")
            return f"Reflection unavailable (HF API Error: {response.status_code}). Check logs for details."
    except requests.exceptions.Timeout:
        print("generate_hf_reflection: Hugging Face API request timed out.")
        return "Reflection unavailable (HF Request Timeout). Check logs for details."
    except Exception as e:
        print(f"generate_hf_reflection: Hugging Face API request failed: {e}")
        traceback.print_exc() # Prints full stack trace to logs
        return "Reflection unavailable (HF Request Exception). Check logs for details."

def generate_gemini_reflection(traits_input: dict) -> str:
    """Generates a reflection using a Google Gemini model."""
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"generate_gemini_reflection: GEMINI_API_KEY is {'SET' if api_key else 'NOT SET'}.")
    if not api_key:
        return "No Gemini API key set. Please check environment variables."
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"generate_gemini_reflection: Error configuring Gemini API: {e}")
        traceback.print_exc()
        return "Gemini reflection unavailable (Configuration Error). Check logs for details."

    default_prompt_text = (
        "Given these soul traits and their values:\n" +
        "\n".join(f"{k}: {v:.2f}" for k, v in traits_input.items()) +
        "\nImagine you are a human soul, alone in infinite darkness, with no sensory input, on the edge of sanity. "
        "Write a short, emotionally raw, subjective, and biased philosophical reflection on this soul's current state, as a human might experience it."
    )
    prompt_to_use = user_prompt if user_prompt else default_prompt_text
    
    print(f"generate_gemini_reflection: Sending request to Gemini with prompt starting: '{prompt_to_use[:70]}...'")

    try:
        model = genai.GenerativeModel("gemini-1.0-flash") # Using the flash model for speed
        response = model.generate_content(prompt_to_use)
        print("generate_gemini_reflection: Successfully received response from Gemini.")
        return response.text.strip()
    except Exception as e:
        print(f"generate_gemini_reflection: Gemini API request failed: {e}")
        # Check for specific feedback from the API if available
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
             print(f"Gemini prompt feedback: {e.response.prompt_feedback}")
        traceback.print_exc()
        return "Gemini reflection unavailable. Check logs for details."

# --- API Endpoints ---

@app.post("/set_prompt")
async def set_prompt_endpoint(request: Request):
    """Allows a user to set a custom prompt for reflections."""
    global user_prompt
    try:
        data = await request.json()
        new_prompt = data.get("prompt")
        if isinstance(new_prompt, str) and new_prompt.strip():
            user_prompt = new_prompt
            print(f"Global user prompt updated to: '{user_prompt}'")
        elif new_prompt is None or not new_prompt.strip():
            user_prompt = None # Reset to default
            print("Global user prompt cleared (will use default).")
        else:
            return JSONResponse({"status": "error", "message": "Invalid prompt format"}, status_code=400)
        return JSONResponse({"status": "ok", "prompt_set_to": user_prompt if user_prompt else "default"})
    except Exception as e:
        print(f"Error in /set_prompt: {e}")
        return JSONResponse({"status": "error", "message": "Invalid request body"}, status_code=400)

# --- Background Simulation Task ---

async def soul_simulation():
    """Runs the main simulation loop in the background."""
    global traits, history, conversation_history, trait_first_seen
    step = 0
    print("Soul simulation loop started.")
    while True:
        # Randomly add a new trait
        if random.random() < 0.07: # 7% chance to add a trait
            new_trait_name = f"trait_{random.randint(1, 10000)}"
            if new_trait_name not in traits:
                traits[new_trait_name] = round(random.random(), 2)
                trait_first_seen[new_trait_name] = step
                print(f"Step {step}: Added new trait '{new_trait_name}' with value {traits[new_trait_name]}")

        # Randomly remove a trait (but keep initial traits and at least 2 total)
        if len(traits) > 2 and random.random() < 0.05: # 5% chance to remove
            # Only remove dynamically added traits
            eligible_to_remove = [k for k in traits if k not in initial_traits] 
            if eligible_to_remove:
                to_remove = random.choice(eligible_to_remove)
                del traits[to_remove]
                if to_remove in trait_first_seen: # Clean up
                    del trait_first_seen[to_remove]
                print(f"Step {step}: Removed trait '{to_remove}'")

        # Evolve all traits
        for k_trait in list(traits.keys()): # Iterate over a copy of keys if modifying dict size
            change = random.uniform(-0.05, 0.05)
            traits[k_trait] = round(min(1.0, max(0.0, traits[k_trait] + change)), 2)
        
        history.append(traits.copy()) # Store a deep copy of current traits
        if len(history) > 200: # Keep history to the last 200 steps
            history = history[-200:]

        # --- PCA and KMeans Analysis (if enough data) ---
        all_traits_list_for_pca = []
        pca_points = []
        clusters = []

        if len(history) >= 3: # Need at least 3 samples for meaningful PCA/KMeans
            # Get a sorted list of all unique trait names that have appeared in history
            all_traits_set = set(k for snap in history for k in snap)
            all_traits_list_for_pca = sorted(list(all_traits_set))
            
            if all_traits_list_for_pca and len(history) >= 3:
                # Create data matrix X: rows are history steps, columns are traits (0 if trait absent)
                X_list = [[snap.get(k_trait, 0.0) for k_trait in all_traits_list_for_pca] for snap in history]
                X = np.array(X_list)

                # Ensure enough samples and features for PCA/KMeans
                if X.shape[0] >= 3 and X.shape[1] > 0:
                    # PCA: Reduce to 2 dimensions if possible
                    n_pca_components = min(2, X.shape[0], X.shape[1]) 
                    if n_pca_components >= 1: # PCA needs at least 1 component
                        try:
                            pca = PCA(n_components=n_pca_components)
                            pca_points = pca.fit_transform(X).tolist()
                        except Exception as e_pca:
                            print(f"Step {step}: PCA error: {e_pca}")
                            # pca_points will remain empty on error
                            
                    # KMeans: Cluster into up to 4 groups
                    n_kmeans_clusters = min(4, X.shape[0]) # Cannot have more clusters than samples
                    if n_kmeans_clusters >= 1: # KMeans needs at least 1 cluster
                        try:
                            kmeans = KMeans(n_clusters=n_kmeans_clusters, n_init='auto', random_state=0)
                            clusters = kmeans.fit_predict(X).tolist()
                        except Exception as e_kmeans:
                            print(f"Step {step}: KMeans error: {e_kmeans}")
                            # clusters will remain empty on error
        
        # --- Generate Reflection ---
        reflection_text = "Reflection generation is pending..."
        speaker_name = "System"
        
        current_traits_for_reflection = traits.copy() # Use a copy for thread safety if needed (though asyncio is single-threaded)
        if step % 2 == 0:
            print(f"Step {step}: Requesting reflection from Hugging Face.")
            reflection_text = generate_hf_reflection(current_traits_for_reflection)
            speaker_name = "AI1 (HF)"
        else:
            print(f"Step {step}: Requesting reflection from Gemini.")
            reflection_text = generate_gemini_reflection(current_traits_for_reflection)
            speaker_name = "AI2 (Gemini)"
        
        print(f"Step {step}: Reflection from {speaker_name} (first 70 chars): '{reflection_text[:70]}...'")
        
        conversation_history.append({
            "speaker": speaker_name,
            "text": reflection_text,
            "timestamp": datetime.now(timezone.utc).isoformat() # Use UTC with timezone info
        })
        if len(conversation_history) > 100: # Keep last 100 conversation entries
            conversation_history = conversation_history[-100:]

        # --- Broadcast Data to Clients ---
        data_to_send = {
            "traits": traits,
            "trait_names": list(traits.keys()), # Current active trait names
            "history": history, # Full history of trait dictionaries
            "step": step,
            "pca_points": pca_points, # PCA projection of history
            "clusters": clusters, # Cluster assignments for history points
            "all_traits_pca_order": all_traits_list_for_pca, # Order of traits used for PCA matrix columns
            "reflection": reflection_text, # The latest reflection text
            "conversation_history": conversation_history,
            "trait_first_seen": trait_first_seen # Step number when each trait first appeared
        }
        
        active_clients = list(clients) # Iterate over a copy in case 'clients' set changes during iteration
        if not active_clients and step % 10 == 0: # Log occasionally if no clients
             print(f"Step {step}: No connected clients to send data to.")
        for ws_client in active_clients:
            try:
                await ws_client.send_json(data_to_send)
            except Exception as e_ws:
                print(f"Step {step}: Error sending data to WebSocket client: {e_ws}. Removing client.")
                clients.discard(ws_client) # Use discard to avoid error if already removed
        
        step += 1
        await asyncio.sleep(5) # Interval between simulation steps (e.g., 5 seconds)

# --- FastAPI Event Handlers and WebSocket ---

@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup."""
    print("Application startup: Initializing soul simulation background task...")
    asyncio.create_task(soul_simulation())
    print("Soul simulation background task created and started.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections."""
    await websocket.accept()
    clients.add(websocket)
    client_host = websocket.client.host if websocket.client else "Unknown host"
    client_port = websocket.client.port if websocket.client else "N/A port"
    print(f"Client {client_host}:{client_port} connected. Total clients: {len(clients)}")
    try:
        while True:
            # Keep connection alive. FastAPI handles PING/PONG.
            # You can receive messages here: text = await websocket.receive_text()
            await asyncio.sleep(60) # Check connection status less frequently
    except Exception: 
        # This block executes on client disconnection or WebSocket errors
        pass 
    finally:
        clients.discard(websocket)
        print(f"Client {client_host}:{client_port} disconnected. Total clients: {len(clients)}")

# --- Static File Serving (for a simple frontend) ---
static_dir_name = "static" # Name of the directory for static files
if not os.path.exists(static_dir_name):
    print(f"Static directory '{static_dir_name}' not found. Creating it.")
    os.makedirs(static_dir_name)
    # Create a basic index.html if it doesn't exist, for quick testing
    dummy_html_path = os.path.join(static_dir_name, "index.html")
    if not os.path.exists(dummy_html_path):
        with open(dummy_html_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soul Simulation Monitor</title>
    <style> body { font-family: sans-serif; margin: 20px; } pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; max-height: 600px; overflow-y: auto; } </style>
</head>
<body>
    <h1>Soul Simulation Monitor 🚀</h1>
    <p>WebSocket data will appear below. Open your browser's developer console (F12) to see detailed JSON objects.</p>
    <pre id="data_display">Connecting to WebSocket...</pre>
    <script>
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${location.host}/ws`);
        const dataDisplay = document.getElementById('data_display');
        ws.onopen = () => { console.log('WebSocket connected!'); dataDisplay.textContent = 'WebSocket connected. Waiting for data...'; };
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Data received:', data);
            dataDisplay.textContent = JSON.stringify(data, null, 2); // Pretty print JSON
        };
        ws.onclose = (event) => { 
            console.log('WebSocket disconnected:', event.reason || 'No reason provided'); 
            dataDisplay.textContent = `WebSocket disconnected. Code: ${event.code}, Reason: ${event.reason || 'N/A'}. Attempting to reconnect in 5s...`;
            setTimeout(() => { window.location.reload(); }, 5000); // Simple reconnect by reloading
        };
        ws.onerror = (error) => { console.error('WebSocket error:', error); dataDisplay.textContent = 'WebSocket error. See console.'; };
    </script>
</body>
</html>""")
        print(f"Created a basic index.html in '{static_dir_name}'.")

app.mount("/", StaticFiles(directory=static_dir_name, html=True), name="static")

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    
    # Assumes your Python file is named "chatbot.py" as per your Render settings.
    # If you rename your file, change "chatbot" to your actual filename (without .py).
    module_name = "chatbot" 
    
    # Render sets the PORT environment variable. Default to 10000 for local consistency.
    port = int(os.getenv("PORT", 10000)) 
    
    print(f"🚀 Starting Uvicorn server on http://0.0.0.0:{port}")
    print(f"👉 Uvicorn will run the FastAPI 'app' instance from '{module_name}.py'")
    print("🔑 Ensure HF_API_KEY and GEMINI_API_KEY are set in your environment.")
    print("📝 Your Render Start Command: uvicorn chatbot.app --host 0.0.0.0 --port 10000")
    
    # For local development, reload=True is helpful. Render usually handles this differently.
    # Render's start command does not use --reload, so we only use it for local __main__ runs.
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=port, reload=True)