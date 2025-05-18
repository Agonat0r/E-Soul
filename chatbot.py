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
import google.api_core.exceptions # To specifically catch NotFound and InvalidArgument
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_COLORS_PALETTE = [
    "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#6A5ACD", 
    "#FF69B4", "#00CED1", "#FFA07A", "#9370DB", "#3CB371",
    "#F08080", "#ADD8E6", "#90EE90", "#FFFFE0", "#C8A2C8",
    "#DB7093", "#AFEEEE", "#F5DEB3", "#DDA0DD", "#8FBC8F",
    "#FA8072", "#B0C4DE", "#98FB98", "#FAFAD2", "#E6E6FA" 
]
DEFAULT_TRAIT_COLOR = "#B0B0B0"

initial_traits = {"empathy": 0.5, "curiosity": 0.5, "self_awareness": 0.5}
traits = initial_traits.copy()
history = []
conversation_history = []
trait_first_seen = {k: 0 for k in initial_traits}

clients = set()
user_prompt = None
user_gemini_api_key = None # Will store API key if set via endpoint

def calculate_trait_similarities_and_colors(
    trait_names_with_history: list[str], 
    trait_history_matrix: np.ndarray, 
    similarity_threshold: float = 0.95,
    min_history_steps_for_similarity: int = 3,
    colors_palette: list[str] = None
) -> dict[str, str]:
    if colors_palette is None: colors_palette = DEFAULT_COLORS_PALETTE
    initial_colors = {name: colors_palette[i % len(colors_palette)] for i, name in enumerate(trait_names_with_history)}
    if (not trait_names_with_history or len(trait_names_with_history) < 2 or 
        not isinstance(trait_history_matrix, np.ndarray) or trait_history_matrix.ndim != 2 or 
        trait_history_matrix.shape[1] != len(trait_names_with_history) or 
        trait_history_matrix.shape[0] < min_history_steps_for_similarity):
        return initial_colors
    try:
        sim_matrix = cosine_similarity(trait_history_matrix.T)
    except Exception as e:
        print(f"Cosine similarity error: {e}"); traceback.print_exc(); return initial_colors
    
    assigned_mask = [False] * len(trait_names_with_history)
    color_idx, grouped_colors = 0, {}
    for i in range(len(trait_names_with_history)):
        if assigned_mask[i]: continue
        current_color = colors_palette[color_idx % len(colors_palette)]
        grouped_colors[trait_names_with_history[i]] = current_color
        assigned_mask[i] = True
        for j in range(i + 1, len(trait_names_with_history)):
            if not assigned_mask[j] and sim_matrix[i, j] >= similarity_threshold:
                grouped_colors[trait_names_with_history[j]] = current_color
                assigned_mask[j] = True
        color_idx += 1
    final_colors = initial_colors.copy(); final_colors.update(grouped_colors)
    return final_colors

def generate_gemini_reflection(traits_input: dict) -> str:
    api_key_to_use = user_gemini_api_key or os.getenv("GEMINI_API_KEY")
    
    # This print statement is crucial for debugging API key issues
    print(f"generate_gemini_reflection: Using API key. User-supplied key is {'PRESENT' if user_gemini_api_key else 'NOT supplied'}. Fallback (env) key is {'PRESENT' if os.getenv('GEMINI_API_KEY') else 'NOT set'}.")

    if not api_key_to_use:
        print("generate_gemini_reflection: CRITICAL - No API key available (neither user-supplied nor environment variable).")
        return "No Gemini API key available (neither user-supplied nor environment variable)."
    
    try:
        genai.configure(api_key=api_key_to_use)
    except Exception as e:
        print(f"generate_gemini_reflection: CRITICAL - Error configuring Gemini API with key ...{api_key_to_use[-4:] if api_key_to_use else 'N/A'}: {e}")
        traceback.print_exc()
        return f"Gemini reflection unavailable (API Key Configuration Error). Check server logs."

    default_prompt_text = (
        "Given these soul traits and their values:\n" +
        "\n".join(f"{k}: {v:.2f}" for k, v in traits_input.items()) +
        "\nImagine you are a human soul, alone in infinite darkness, with no sensory input, on the edge of sanity. "
        "You have a lifespan of potentially 1000 steps. "
        "Write a short, emotionally raw, subjective, and biased philosophical reflection on this soul's current state, as a human might experience it."
    )
    prompt_to_use = user_prompt if user_prompt else default_prompt_text
    
    model_name_being_used = "models/gemini-2.0-flash" # As per user's existing code
    print(f"generate_gemini_reflection: Attempting to use Gemini model: '{model_name_being_used}' with key ...{api_key_to_use[-4:] if api_key_to_use else 'N/A'}")

    try:
        model = genai.GenerativeModel(model_name_being_used) 
        response = model.generate_content(prompt_to_use, request_options={"timeout": 20}) # Added timeout
        
        if hasattr(response, 'text') and response.text and response.text.strip():
            print(f"generate_gemini_reflection: Successfully received response from model '{model_name_being_used}'.")
            return response.text.strip()
        else:
            block_reason_msg = ""
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = f" (Block Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason})"
            error_message = f"Gemini reflection unavailable (Empty or Blocked response from model '{model_name_being_used}'{block_reason_msg})."
            print(f"generate_gemini_reflection: {error_message}")
            if hasattr(response, 'prompt_feedback'): print(f"Prompt Feedback: {response.prompt_feedback}")
            if hasattr(response, 'candidates') and response.candidates: print(f"Candidates: {response.candidates}")
            return error_message

    except google.api_core.exceptions.NotFound as e_nf:
        error_message = f"Gemini reflection unavailable (Model '{model_name_being_used}' NOT FOUND: {e_nf}). Verify model name via list_models()."
        print(f"generate_gemini_reflection: CRITICAL - {error_message}")
        traceback.print_exc()
        return error_message
    except google.api_core.exceptions.InvalidArgument as e_ia: 
        error_message = f"Gemini reflection unavailable (INVALID API KEY or Argument: {e_ia}). Check API key & server logs."
        print(f"generate_gemini_reflection: CRITICAL - {error_message}")
        traceback.print_exc()
        return error_message
    except requests.exceptions.Timeout: # More specific timeout for genai, though less common
        error_message = f"Gemini reflection unavailable (Request TIMEOUT for model '{model_name_being_used}')."
        print(f"generate_gemini_reflection: {error_message}")
        traceback.print_exc()
        return error_message
    except Exception as e:
        error_message = f"Gemini reflection unavailable (Unexpected Error: {e}). Check server logs."
        print(f"generate_gemini_reflection: ERROR - {error_message}")
        traceback.print_exc()
        return error_message

@app.post("/set_prompt")
async def set_prompt_endpoint(request: Request):
    global user_prompt; data = await request.json(); new_prompt = data.get("prompt")
    if isinstance(new_prompt, str) and new_prompt.strip(): user_prompt = new_prompt; print(f"User prompt set: '{user_prompt}'")
    elif new_prompt is None or (isinstance(new_prompt, str) and not new_prompt.strip()): user_prompt = None; print("User prompt cleared.")
    else: return JSONResponse({"status":"error","message":"Invalid prompt format"},status_code=400)
    return JSONResponse({"status":"ok","prompt_set_to":user_prompt or "default"})

@app.post("/set_api_keys")
async def set_api_keys(request: Request):
    global user_gemini_api_key # Only managing Gemini key as per user's Python script
    data = await request.json()
    gemini_key_from_request = data.get("gemini_api_key")
    # hf_key_from_request = data.get("hf_api_key") # Not used in current Python backend
    errors = {}
    
    print(f"/set_api_keys: Received request. Gemini key provided: {'Yes' if gemini_key_from_request else 'No'}")

    if gemini_key_from_request:
        try:
            print(f"/set_api_keys: Validating provided Gemini key ending in ...{gemini_key_from_request[-4:]}")
            # Temporarily configure to test this specific key
            # Note: genai.configure is global. This will affect other calls if not careful.
            # However, generate_gemini_reflection re-configures each time.
            genai.configure(api_key=gemini_key_from_request)
            
            model_to_test_with = "models/gemini-2.0-flash" # Same model as main function
            model = genai.GenerativeModel(model_to_test_with)
            
            print(f"/set_api_keys: Making test call to model '{model_to_test_with}'...")
            test_response = model.generate_content("Test", request_options={"timeout": 10}) # Short timeout for test
            
            if hasattr(test_response, 'text') and test_response.text and test_response.text.strip():
                user_gemini_api_key = gemini_key_from_request
                print(f"/set_api_keys: Gemini API key ending in ...{user_gemini_api_key[-4:]} VALIDATED and SET successfully.")
            else:
                block_reason_msg = ""
                if hasattr(test_response, 'prompt_feedback') and test_response.prompt_feedback and test_response.prompt_feedback.block_reason:
                    block_reason_msg = f" (Block Reason: {test_response.prompt_feedback.block_reason_message or test_response.prompt_feedback.block_reason})"
                errors["gemini_api_key"] = f"Invalid Gemini API key (Test call to '{model_to_test_with}' returned empty/blocked response{block_reason_msg})."
                print(f"/set_api_keys: Validation FAILED for key ...{gemini_key_from_request[-4:]}. {errors['gemini_api_key']}")
                if hasattr(test_response, 'prompt_feedback'): print(f"Test Call Prompt Feedback: {test_response.prompt_feedback}")
        
        except google.api_core.exceptions.NotFound as e_nf:
            errors["gemini_api_key"] = f"Gemini API key validation error: Model '{model_to_test_with}' NOT FOUND. {e_nf}"
            print(f"/set_api_keys: Validation FAILED for key ...{gemini_key_from_request[-4:]}. {errors['gemini_api_key']}")
            traceback.print_exc()
        except google.api_core.exceptions.InvalidArgument as e_ia:
            errors["gemini_api_key"] = f"Gemini API key validation error: INVALID KEY or Argument. {e_ia}"
            print(f"/set_api_keys: Validation FAILED for key ...{gemini_key_from_request[-4:]}. {errors['gemini_api_key']}")
            traceback.print_exc()
        except Exception as e: # Catch any other exception during validation
            errors["gemini_api_key"] = f"Gemini API key validation error: Unexpected exception. {str(e)}"
            print(f"/set_api_keys: Validation FAILED for key ...{gemini_key_from_request[-4:]}. {errors['gemini_api_key']}")
            traceback.print_exc()
    else:
        errors["gemini_api_key"] = "No Gemini API key provided in request."
        print(f"/set_api_keys: {errors['gemini_api_key']}")

    if errors:
        user_gemini_api_key = None # Ensure key is not set if validation fails
        return JSONResponse({"status": "error", "errors": errors}, status_code=400)
    return JSONResponse({"status": "ok", "message": "Gemini API key accepted and set."})


async def soul_simulation():
    global traits, history, conversation_history, trait_first_seen
    step = 0; initial_trait_set = set(initial_traits.keys()); print("Soul simulation loop started.")
    while True:
        for k in traits: traits[k] = round(min(1.0, max(0.0, traits[k] + random.uniform(-0.05, 0.05))),2)
        history.append(traits.copy());
        if len(history) > 200: history=history[-200:]
        
        aptl,pcp,cl,X=[],[],[],np.array([])
        mhfa=3 # Min history for analysis
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
        
        ctna=list(traits.keys());ftc={}
        cfs={}
        if aptl and X.ndim==2 and X.shape[0]>=1 and X.shape[1]==len(aptl):
            cfs=calculate_trait_similarities_and_colors(trait_names_with_history=aptl,trait_history_matrix=X,min_history_steps_for_similarity=mhfa)
        ftc.update(cfs)
        uc=set(ftc.values());ap=[c for c in DEFAULT_COLORS_PALETTE if c not in uc]
        if not ap:ap=DEFAULT_COLORS_PALETTE
        cin=0
        for n in ctna:
            if n not in ftc:
                if ap:ftc[n]=ap[cin%len(ap)];cin+=1
                else:ftc[n]=DEFAULT_TRAIT_COLOR
        ftc={n:ftc.get(n,DEFAULT_TRAIT_COLOR)for n in ctna}

        current_traits_snapshot = traits.copy()
        rt = generate_gemini_reflection(current_traits_snapshot)
        sn = "AI (Gemini)"
        # print(f"Step {step}: Main reflection from {sn} (first 70): '{rt[:70]}...'") # Optional: reduce log verbosity
        conversation_history.append({"speaker":sn,"text":rt,"timestamp":datetime.now(timezone.utc).isoformat()})

        api_key_for_extra_calls = user_gemini_api_key or os.getenv("GEMINI_API_KEY")
        model_for_extra_calls_name = "models/gemini-2.0-flash"

        if api_key_for_extra_calls and not rt.startswith("No Gemini API key") and not ("unavailable" in rt.lower() and "gemini" in rt.lower()):
            try:
                genai.configure(api_key=api_key_for_extra_calls)
                model_for_extra_calls = genai.GenerativeModel(model_for_extra_calls_name)
                
                # --- Self-Dialogue ---
                reply_prompt = ("You are continuing a conversation with yourself. Here is your previous message:\n" + rt +
                                "\nReply to yourself, as if you are reflecting further or challenging your own thoughts. Be concise.")
                reply_response = model_for_extra_calls.generate_content(reply_prompt, request_options={"timeout": 15})
                if hasattr(reply_response, 'text') and reply_response.text and reply_response.text.strip():
                    reply_text = reply_response.text.strip()
                    conversation_history.append({"speaker": "AI-SelfReply (Gemini)", "text": reply_text, "timestamp": datetime.now(timezone.utc).isoformat()})
                else:
                    print(f"S{step}: Gemini self-dialogue returned empty/blocked. Feedback: {getattr(reply_response, 'prompt_feedback', 'N/A')}")
            except Exception as e:
                print(f"S{step}: Gemini self-dialogue error: {e}"); # traceback.print_exc() # Can be verbose

            # --- AI-driven Trait Evolution ---
            try: # New try-except for trait evolution
                # genai.configure already called above for this block
                # model_for_extra_calls = genai.GenerativeModel(model_for_extra_calls_name) # Already instantiated
                ai_trait_prompt = (
                    "Current traits:\n" + "\n".join(f"{k}: {v:.2f}" for k, v in current_traits_snapshot.items()) +
                    "\n\nBased on your current state and thoughts (last reflection: '" + rt[:100] + "...'), "
                    "suggest one new trait name (one concise word, e.g., 'despair', 'entropy', 'hope') that could describe your soul's evolution. "
                    "Also, if you feel any *existing non-initial* trait is no longer relevant or redundant, name one for removal. "
                    "Respond ONLY in this format (NO other text):\n"
                    "New trait: <trait_name_or_none>\nRemove trait: <trait_name_or_none>"
                )
                ai_trait_response = model_for_extra_calls.generate_content(ai_trait_prompt, request_options={"timeout": 15})
                if hasattr(ai_trait_response, 'text') and ai_trait_response.text and ai_trait_response.text.strip():
                    ai_trait_text = ai_trait_response.text.strip()
                    new_trait_candidate, remove_trait_candidate = None, None
                    for line in ai_trait_text.splitlines():
                        if line.lower().startswith("new trait:"): new_trait_candidate = line.split(":", 1)[1].strip().lower()
                        if line.lower().startswith("remove trait:"): remove_trait_candidate = line.split(":", 1)[1].strip().lower()
                    if new_trait_candidate and new_trait_candidate!='none' and new_trait_candidate not in traits and len(new_trait_candidate.split())==1:
                        traits[new_trait_candidate] = round(random.random(), 2); trait_first_seen[new_trait_candidate] = step
                        print(f"S{step}: AI NEW trait '{new_trait_candidate}' added.")
                    if remove_trait_candidate and remove_trait_candidate!='none' and remove_trait_candidate in traits and remove_trait_candidate not in initial_trait_set:
                        del traits[remove_trait_candidate]; trait_first_seen.pop(remove_trait_candidate, None)
                        print(f"S{step}: AI REMOVAL of trait '{remove_trait_candidate}'.")
                else:
                    print(f"S{step}: AI trait evolution returned empty/blocked. Feedback: {getattr(ai_trait_response, 'prompt_feedback', 'N/A')}")
            except Exception as e:
                print(f"S{step}: AI trait evolution error: {e}"); # traceback.print_exc()
        else:
            if not api_key_for_extra_calls: print(f"S{step}: Skipping self-dialogue/trait-evo, no API key.")
            else: print(f"S{step}: Skipping self-dialogue/trait-evo due to main reflection failure/issue ('{rt[:30]}...').")


        if len(conversation_history) > 100: conversation_history=conversation_history[-100:]
        
        data_to_send={"traits":traits,"trait_names":ctna,"history":history,"step":step,"pca_points":pcp,"clusters":cl,
                      "all_traits_pca_order":aptl,"reflection":rt,"conversation_history":conversation_history,
                      "trait_first_seen":trait_first_seen,"trait_colors":ftc}
        active_clients = list(clients)
        if not active_clients and step % 20 == 0 : print(f"S{step}: No clients connected.") # Log if no clients periodically
        for wc in active_clients:
            try:await wc.send_json(data_to_send)
            except:clients.discard(wc) # Remove if send fails
        step += 1
        await asyncio.sleep(7) 

@app.on_event("startup")
async def startup_event():asyncio.create_task(soul_simulation());print("Soul sim task created.")
@app.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    await websocket.accept();clients.add(websocket);ch=websocket.client.host or "Unk";cp=websocket.client.port or "N/A"
    print(f"Client {ch}:{cp} connected. Total clients:{len(clients)}")
    try:
        while True:await asyncio.sleep(60) # Keep alive, prevent timeout
    except:pass # Handle client disconnect
    finally:clients.discard(websocket);print(f"Client {ch}:{cp} disconnected. Total clients:{len(clients)}")

static_dir_name="static"
if not os.path.exists(static_dir_name):
    os.makedirs(static_dir_name)
    dummy_html_path=os.path.join(static_dir_name,"index.html") # Using user's provided HTML now
    if not os.path.exists(dummy_html_path):
        # Create the more advanced HTML if it doesn't exist
        # For brevity, I will assume the user has their HTML file named index.html in the static folder
        # If not, they can copy their HTML content into static/index.html
        print(f"NOTE: Please ensure your custom HTML file is placed at '{static_dir_name}/index.html'")
        # Fallback to simpler HTML if user's is not there by default.
        with open(dummy_html_path,"w")as f:f.write("""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Soul Sim (Basic)</title></head><body><h1>Soul Sim Monitor</h1><pre id="data_display">Connecting...</pre>
<script>const ws=new WebSocket(`${window.location.protocol==='https:'?'wss:':'ws:'}//${location.host}/ws`);const dd=document.getElementById('data_display');ws.onopen=()=>{dd.textContent='Connected.';};ws.onmessage=(evt)=>{const d=JSON.parse(evt.data);console.log('Data:',d);dd.textContent=JSON.stringify(d,null,2);};ws.onclose=(evt)=>{dd.textContent='Disconnected.';};ws.onerror=(err)=>{console.error('WS Error:',err);dd.textContent='WS error.';};</script></body></html>""")


app.mount("/",StaticFiles(directory=static_dir_name,html=True),name="static")

if __name__=="__main__":
    import uvicorn;mn="chatbot";p=int(os.getenv("PORT",10000))
    print(f"🚀 Starting Uvicorn server on http://0.0.0.0:{p}")
    print(f"👉 Running FastAPI 'app' from '{mn}.py'")
    print("🔑 Ensure GEMINI_API_KEY (environment variable) is set if not using /set_api_keys, or if submitted key is invalid.")
    uvicorn.run(f"{mn}:app",host="0.0.0.0",port=p,reload=True)