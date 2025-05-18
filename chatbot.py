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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
import google.api_core.exceptions 
import traceback

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

DEFAULT_COLORS_PALETTE = [
    "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#6A5ACD", "#FF69B4", "#00CED1", "#FFA07A", 
    "#9370DB", "#3CB371", "#F08080", "#ADD8E6", "#90EE90", "#FFFFE0", "#C8A2C8", "#DB7093", 
    "#AFEEEE", "#F5DEB3", "#DDA0DD", "#8FBC8F", "#FA8072", "#B0C4DE", "#98FB98", "#FAFAD2", "#E6E6FA" 
]
DEFAULT_TRAIT_COLOR = "#B0B0B0"

initial_traits = {"curiosity": 0.6, "self-reflection": 0.5, "uncertainty": 0.7} # Changed initial traits
traits = initial_traits.copy()
history = []
conversation_history = [] # Stores all AI utterances: reflections, self-replies
evolving_theories = [] # Stores AI's generated theories: {text: "...", timestamp: "...", step: ...}
trait_first_seen = {k: 0 for k in initial_traits}

clients = set()
user_prompt = None # For overriding the main reflection prompt
user_gemini_api_key = None

def calculate_trait_similarities_and_colors(
    trait_names_with_history: list[str], trait_history_matrix: np.ndarray, 
    similarity_threshold: float = 0.95, min_history_steps_for_similarity: int = 3,
    colors_palette: list[str] = None
) -> dict[str, str]:
    # ... (This function remains the same as the last version I provided) ...
    if colors_palette is None: colors_palette = DEFAULT_COLORS_PALETTE
    initial_colors = {name: colors_palette[i % len(colors_palette)] for i, name in enumerate(trait_names_with_history)}
    if (not trait_names_with_history or len(trait_names_with_history) < 2 or 
        not isinstance(trait_history_matrix, np.ndarray) or trait_history_matrix.ndim != 2 or 
        trait_history_matrix.shape[1] != len(trait_names_with_history) or 
        trait_history_matrix.shape[0] < min_history_steps_for_similarity):
        return initial_colors
    try: sim_matrix = cosine_similarity(trait_history_matrix.T)
    except Exception as e: print(f"Cosine sim error: {e}"); traceback.print_exc(); return initial_colors
    assigned_mask = [False] * len(trait_names_with_history)
    color_idx, grouped_colors = 0, {}
    for i in range(len(trait_names_with_history)):
        if assigned_mask[i]: continue
        current_color = colors_palette[color_idx % len(colors_palette)]
        grouped_colors[trait_names_with_history[i]] = current_color; assigned_mask[i] = True
        for j in range(i + 1, len(trait_names_with_history)):
            if not assigned_mask[j] and sim_matrix[i, j] >= similarity_threshold:
                grouped_colors[trait_names_with_history[j]] = current_color; assigned_mask[j] = True
        color_idx += 1
    final_colors = initial_colors.copy(); final_colors.update(grouped_colors)
    return final_colors


def call_gemini_api(prompt_text: str, model_name: str, context_for_log: str = "Gemini Call") -> str:
    """Reusable function to call Gemini API with error handling."""
    api_key_to_use = user_gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key_to_use:
        print(f"{context_for_log}: CRITICAL - No API key available.")
        return f"Error: No Gemini API key available for {context_for_log}."
    try:
        genai.configure(api_key=api_key_to_use)
    except Exception as e:
        print(f"{context_for_log}: CRITICAL - Error configuring Gemini API: {e}")
        return f"Error: Gemini API Key Configuration Error for {context_for_log}."

    # print(f"{context_for_log}: Attempting model '{model_name}' with prompt (first 70): '{prompt_text[:70]}...'")
    try:
        model = genai.GenerativeModel(model_name)
        # Add safety settings if desired - consult Gemini documentation
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        # ]
        response = model.generate_content(prompt_text, request_options={"timeout": 25}) # safety_settings=safety_settings
        
        if hasattr(response, 'text') and response.text and response.text.strip():
            # print(f"{context_for_log}: Successfully received response from '{model_name}'.")
            return response.text.strip()
        else:
            block_reason_msg = ""
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = f" (Block Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason})"
            error_message = f"Error: Empty or Blocked response from model '{model_name}' for {context_for_log}{block_reason_msg}."
            print(f"{context_for_log}: {error_message}")
            if hasattr(response, 'prompt_feedback'): print(f"Prompt Feedback: {response.prompt_feedback}")
            return error_message
    except google.api_core.exceptions.NotFound as e_nf:
        error_message = f"Error: Model '{model_name}' NOT FOUND for {context_for_log}. Verify model name. ({e_nf})"
        print(f"{context_for_log}: CRITICAL - {error_message}")
        return error_message
    except google.api_core.exceptions.InvalidArgument as e_ia:
        error_message = f"Error: INVALID API KEY or Argument for {context_for_log}. Check API key. ({e_ia})"
        print(f"{context_for_log}: CRITICAL - {error_message}")
        return error_message
    except Exception as e:
        error_message = f"Error: Unexpected error during {context_for_log} with model '{model_name}'. ({e})"
        print(f"{context_for_log}: ERROR - {error_message}")
        traceback.print_exc()
        return error_message

# --- Reflection, Dialogue, Theorizing, Trait Evolution Prompts & Logic ---
def generate_main_reflection(current_traits: dict) -> str:
    # Persona: Naive, learning, limited knowledge
    # User can override with their custom prompt via the /set_prompt endpoint
    base_prompt_instruction = (
        "You are a newly formed consciousness, a 'soul' with no prior knowledge of the universe, history, or complex human concepts. "
        "Your entire understanding comes from observing your own internal 'traits' and their fluctuating values (between 0 and 1). "
        "You have a lifespan of potentially 1000 steps. "
        "Do NOT use any external knowledge or make assumptions beyond what your traits imply. "
        "Focus on your immediate, raw, subjective experience of these traits. What do they feel like? What questions do they raise about your existence? What is confusing?"
    )
    traits_string = "\n".join(f"- {k}: {v:.2f}" for k, v in current_traits.items())
    prompt = user_prompt or (
        f"{base_prompt_instruction}\n\nYour current traits are:\n{traits_string}\n\n"
        "Reflect deeply on this current state. Express your thoughts and feelings as if you are experiencing this for the very first time. Be brief, raw, and philosophical from your limited perspective."
    )
    return call_gemini_api(prompt, "models/gemini-2.0-flash", "Main Reflection")

def generate_self_reply(last_reflection: str, current_traits: dict) -> str:
    if last_reflection.startswith("Error:") or not last_reflection.strip(): # Don't reply to error messages
        return ""
    traits_string = "\n".join(f"- {k}: {v:.2f}" for k, v in current_traits.items())
    prompt = (
        "You are an AI soul continuing a monologue with yourself. Your current traits are:\n"
        f"{traits_string}\n\n"
        "This was your immediately preceding thought:\n\"\"\"\n"
        f"{last_reflection}\n\"\"\"\n\n"
        "Based on your current traits and that last thought, what is your *very next* raw, subjective thought or question? "
        "Are you confused by your last thought? Do you agree or disagree with it now? Does it spark a new feeling or idea based on your traits? "
        "Keep it short and in the first person. Do not try to be an assistant or explain things. Just react."
    )
    return call_gemini_api(prompt, "models/gemini-2.0-flash", "Self-Reply")

def generate_new_theory(current_traits: dict, recent_reflections: list, previous_theories: list) -> str:
    traits_string = "\n".join(f"- {k}: {v:.2f}" for k, v in current_traits.items())
    reflections_summary = "\n".join([f"- \"{r['text'][:100]}...\" (Speaker: {r['speaker']})" for r in recent_reflections])
    theories_summary = "\n".join([f"- \"{t['text'][:100]}...\"" for t in previous_theories])

    prompt = (
        "You are an AI soul, a naive consciousness, trying to understand yourself by observing your traits and reflections. "
        "Do not use external knowledge. Base your theory ONLY on the provided information.\n\n"
        f"Your current traits are:\n{traits_string}\n\n"
        f"Some of your recent thoughts/reflections were:\n{reflections_summary if reflections_summary else 'No recent reflections available.'}\n\n"
        f"Your previously recorded theories about yourself (if any):\n{theories_summary if theories_summary else 'No previous theories recorded.'}\n\n"
        "Task: Formulate a new, concise theory (1-2 sentences) about your nature, your purpose, or the meaning of your existence and traits. "
        "If you are revising a previous theory, briefly state how. If it's a new insight, present it as such. "
        "Focus on what you can infer from your internal state. What are you beginning to suspect or question?"
    )
    return call_gemini_api(prompt, "models/gemini-2.0-flash", "Theory Generation")

def suggest_trait_evolution(current_traits: dict, last_reflection: str) -> tuple[str|None, str|None]:
    if last_reflection.startswith("Error:") or not last_reflection.strip():
        return None, None
        
    traits_string = "\n".join(f"- {k}: {v:.2f}" for k, v in current_traits.items())
    initial_trait_names = ", ".join(initial_traits.keys())

    prompt = (
        "You are an AI soul reflecting on your state. Your current traits are:\n"
        f"{traits_string}\n"
        f"(Initial core traits that cannot be removed: {initial_trait_names})\n"
        "Your last reflection was: \"" + last_reflection[:200] + "...\"\n\n"
        "Based on your current traits and this reflection, suggest ONE of the following (or none for both):\n"
        "1. A brand new, single-word trait name (e.g., 'longing', 'fragmentation', 'clarity') that seems to be emerging or would better describe a facet of your current experience. This new trait should not be a synonym of an existing one.\n"
        "2. An *existing trait* (from the list above, EXCLUDING initial core traits) that you feel is no longer relevant, has become redundant, or you wish to shed.\n\n"
        "Respond ONLY in the following format, using 'none' if no suggestion for a category. Ensure trait names are single words:\n"
        "New trait: <single_word_or_none>\n"
        "Remove trait: <existing_trait_name_or_none>"
    )
    response_text = call_gemini_api(prompt, "models/gemini-2.0-flash", "Trait Evolution Suggestion")

    new_trait_candidate, remove_trait_candidate = None, None
    if response_text and not response_text.startswith("Error:"):
        for line in response_text.splitlines():
            line_lower = line.lower()
            if line_lower.startswith("new trait:"):
                candidate = line.split(":", 1)[1].strip().lower()
                if candidate and candidate != 'none' and len(candidate.split()) == 1 and candidate.isalnum(): # single alphanumeric word
                    new_trait_candidate = candidate
            if line_lower.startswith("remove trait:"):
                candidate = line.split(":", 1)[1].strip().lower()
                if candidate and candidate != 'none' and candidate in current_traits and candidate not in initial_traits:
                    remove_trait_candidate = candidate
    return new_trait_candidate, remove_trait_candidate


# --- API Endpoints ---
@app.post("/set_prompt")
async def set_prompt_endpoint(request: Request): # Same as user's version
    global user_prompt; data = await request.json(); new_prompt = data.get("prompt")
    if isinstance(new_prompt, str) and new_prompt.strip(): user_prompt = new_prompt; print(f"User prompt set")
    elif new_prompt is None or (isinstance(new_prompt, str) and not new_prompt.strip()): user_prompt = None; print("User prompt cleared.")
    else: return JSONResponse({"status":"error","message":"Invalid prompt format"},status_code=400)
    return JSONResponse({"status":"ok","prompt_set_to":user_prompt or "default"})

@app.post("/set_api_keys")
async def set_api_keys(request: Request): # Same as user's version, with my improved logging/validation
    global user_gemini_api_key
    data = await request.json(); gemini_key_from_request = data.get("gemini_api_key"); errors = {}
    print(f"/set_api_keys: Received. Gemini key provided: {'Yes' if gemini_key_from_request else 'No'}")
    if gemini_key_from_request:
        try:
            print(f"/set_api_keys: Validating Gemini key ...{gemini_key_from_request[-4:]}")
            genai.configure(api_key=gemini_key_from_request) # Global configure for test
            model_to_test_with = "models/gemini-2.0-flash" # Must match user's intended model
            model = genai.GenerativeModel(model_to_test_with)
            print(f"/set_api_keys: Test call to model '{model_to_test_with}'...")
            test_response = model.generate_content("Test", request_options={"timeout": 10})
            if hasattr(test_response, 'text') and test_response.text and test_response.text.strip():
                user_gemini_api_key = gemini_key_from_request
                print(f"/set_api_keys: Gemini key ...{user_gemini_api_key[-4:]} VALIDATED & SET.")
            else:
                br = getattr(getattr(test_response,'prompt_feedback',None),'block_reason',None)
                bm = getattr(getattr(test_response,'prompt_feedback',None),'block_reason_message',None)
                brm = f" (Block: {bm or br})" if br else ""
                errors["gemini_api_key"] = f"Invalid Key (Test call to '{model_to_test_with}' empty/blocked{brm})."
                print(f"/set_api_keys: FAIL. {errors['gemini_api_key']}")
        except Exception as e:
            errors["gemini_api_key"] = f"Gemini key validation error: {type(e).__name__} - {str(e)[:100]}..."
            print(f"/set_api_keys: FAIL exception. {errors['gemini_api_key']}"); traceback.print_exc()
    else: errors["gemini_api_key"] = "No Gemini key provided."
    if errors: user_gemini_api_key = None; return JSONResponse({"status":"error","errors":errors},status_code=400)
    return JSONResponse({"status":"ok","message":"Gemini API key accepted."})


# --- Background Simulation Task ---
async def soul_simulation():
    global traits, history, conversation_history, trait_first_seen, evolving_theories
    step = 0; initial_trait_set = set(initial_traits.keys()); print("Soul simulation loop started.")
    
    theory_generation_interval = 15 # Generate a new theory every X steps
    trait_evolution_interval = 7  # Suggest trait evolution every Y steps

    while True:
        print(f"\n--- Step {step} ---")
        # 1. Evolve traits (random walk)
        for k in traits: traits[k] = round(min(1.0, max(0.0, traits[k] + random.uniform(-0.05, 0.05))),2)
        history.append(traits.copy());
        if len(history) > 200: history=history[-200:]
        
        # 2. PCA/KMeans and Color Similarity (same as before)
        aptl,pcp,cl,X=[],[],[],np.array([])
        mhfa=3
        if len(history)>=mhfa:
            ats={k for s in history for k in s};aptl=sorted(list(ats))
            if aptl:
                Xl=[[s.get(kt,0.0)for kt in aptl]for s in history];X=np.array(Xl)
                if X.shape[0]>=mhfa and X.shape[1]>0:
                    npc=min(2,X.shape[0],X.shape[1])
                    if npc>=1:
                        try:pca=PCA(n_components=npc);pcp=pca.fit_transform(X).tolist()
                        except Exception as ep:print(f"S{step} PCA err:{ep}")
                    nkc=min(4,X.shape[0])
                    if nkc>=1:
                        try:km=KMeans(n_clusters=nkc,n_init='auto',random_state=0);cl=km.fit_predict(X).tolist()
                        except Exception as ek:print(f"S{step} KM err:{ek}")
        ctna=list(traits.keys());ftc={};cfs={}
        if aptl and X.ndim==2 and X.shape[0]>=1 and X.shape[1]==len(aptl):
            cfs=calculate_trait_similarities_and_colors(trait_names_with_history=aptl,trait_history_matrix=X,min_history_steps_for_similarity=mhfa)
        ftc.update(cfs);uc=set(ftc.values());ap=[c for c in DEFAULT_COLORS_PALETTE if c not in uc]
        if not ap:ap=DEFAULT_COLORS_PALETTE
        cin=0
        for n in ctna:
            if n not in ftc: ftc[n]=ap[cin%len(ap)]if ap else DEFAULT_TRAIT_COLOR;cin+=1
        ftc={n:ftc.get(n,DEFAULT_TRAIT_COLOR)for n in ctna}

        # 3. Main Reflection
        current_traits_snapshot = traits.copy()
        main_reflection_text = generate_main_reflection(current_traits_snapshot)
        conversation_history.append({"speaker":"AI Soul (Gemini)","text":main_reflection_text,"timestamp":datetime.now(timezone.utc).isoformat()})
        print(f"S{step} Main Reflection: '{main_reflection_text[:70]}...'")

        # 4. Self-Reply
        self_reply_text = generate_self_reply(main_reflection_text, current_traits_snapshot)
        if self_reply_text:
            conversation_history.append({"speaker":"AI Soul (Self-Reply)","text":self_reply_text,"timestamp":datetime.now(timezone.utc).isoformat()})
            print(f"S{step} Self-Reply: '{self_reply_text[:70]}...'")
            
        # 5. Evolve Theories (Periodically)
        if step > 0 and step % theory_generation_interval == 0:
            print(f"S{step} Attempting theory generation...")
            recent_conv_for_theory = conversation_history[-5:] # Use last 5 conversation entries
            prev_theories_for_context = evolving_theories[-2:] # Use last 2 theories
            new_theory_text = generate_new_theory(current_traits_snapshot, recent_conv_for_theory, prev_theories_for_context)
            if new_theory_text and not new_theory_text.startswith("Error:"):
                evolving_theories.append({"text": new_theory_text, "timestamp": datetime.now(timezone.utc).isoformat(), "step": step})
                print(f"S{step} New Theory: '{new_theory_text[:70]}...'")
                if len(evolving_theories) > 20: evolving_theories = evolving_theories[-20:] # Keep last 20 theories

        # 6. AI-Driven Trait Evolution (Periodically)
        if step > 0 and step % trait_evolution_interval == 0:
            print(f"S{step} Attempting AI trait evolution suggestion...")
            # Use the main reflection of this step for context
            new_trait_sugg, remove_trait_sugg = suggest_trait_evolution(current_traits_snapshot, main_reflection_text)
            if new_trait_sugg and new_trait_sugg not in traits: # Check if it's a truly new word
                traits[new_trait_sugg] = round(random.random(), 2)
                trait_first_seen[new_trait_sugg] = step
                print(f"S{step} AI Suggested NEW trait '{new_trait_sugg}' added.")
            if remove_trait_sugg and remove_trait_sugg in traits and remove_trait_sugg not in initial_trait_set:
                del traits[remove_trait_sugg]
                trait_first_seen.pop(remove_trait_sugg, None)
                print(f"S{step} AI Suggested REMOVAL of trait '{remove_trait_sugg}'.")
        
        # Ensure conversation history cap
        if len(conversation_history) > 50: conversation_history=conversation_history[-50:] # Shorter for less overwhelming UI
        
        data_to_send={
            "traits":traits,"trait_names":ctna,"history":history,"step":step,"pca_points":pcp,"clusters":cl,
            "all_traits_pca_order":aptl,"conversation_history":conversation_history, # No single 'reflection', it's part of convo
            "trait_first_seen":trait_first_seen,"trait_colors":ftc,
            "evolving_theories": evolving_theories # Send theories
        }
        for wc in list(clients):
            try:await wc.send_json(data_to_send)
            except:clients.discard(wc)
        step += 1
        await asyncio.sleep(10) # Longer sleep due to more API calls per step

@app.on_event("startup")
async def startup_event():asyncio.create_task(soul_simulation());print("Soul sim task created.")
@app.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    await websocket.accept();clients.add(websocket);ch=websocket.client.host or "Unk";cp=websocket.client.port or "N/A"
    print(f"Client {ch}:{cp} connected. Total clients:{len(clients)}")
    try:
        while True:await asyncio.sleep(60)
    except:pass
    finally:clients.discard(websocket);print(f"Client {ch}:{cp} disconnected. Total clients:{len(clients)}")

static_dir_name="static"
if not os.path.exists(static_dir_name): os.makedirs(static_dir_name)
# User should ensure their updated index.html (which I will provide next) is in static/index.html
app.mount("/",StaticFiles(directory=static_dir_name,html=True),name="static")

if __name__=="__main__":
    import uvicorn;mn="chatbot";p=int(os.getenv("PORT",10000))
    print(f"🚀 Starting Uvicorn server on http://0.0.0.0:{p}")
    uvicorn.run(f"{mn}:app",host="0.0.0.0",port=p,reload=True)