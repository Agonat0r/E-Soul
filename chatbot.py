import asyncio
import random
import os
import requests # Retained for potential future use, not strictly needed if only Gemini
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

# --- Evolutionary Stages & Trait Concepts ---
EVOLUTIONARY_STAGES = [
    {"name": "Primordial Essence", "description": "A pre-sentient state, basic reactivity and energy absorption."},
    {"name": "Microorganism", "description": "Focus on basic survival: energy processing, replication drive, simple stimulus-response, basic motility."},
    {"name": "Colonial Organism", "description": "Rudimentary cell cooperation, basic resource sharing, simple environmental sensing, early defense."},
    {"name": "Instinctual Creature", "description": "Development of core instincts, rudimentary emotions, goal-directed behavior (food, safety, reproduction)."},
    {"name": "Developing Mind", "description": "Emergence of complex emotions, social bonds, learning from experience, basic problem-solving, heightened curiosity."},
    {"name": "Proto-Sapient", "description": "Early abstract thought, basic tool use/environmental modification, complex communication, deepening self-awareness."},
    {"name": "Sapient Being", "description": "Metacognition, forming complex theories of self and universe, foresight, ethical considerations, creativity."}
]
STAGE_RELEVANT_TRAIT_CONCEPTS = {
    "Primordial Essence": ["reactivity", "energy-uptake", "stability"],
    "Microorganism": ["metabolism", "motility", "replication-drive", "chemo-sensitivity", "resilience"],
    "Colonial Organism": ["cell-adhesion", "group-synchrony", "external-sensing", "defense-mechanism", "resource-distribution"],
    "Instinctual Creature": ["fear-response", "aggression", "hunger-drive", "mating-urge", "basic-memory", "pain-avoidance", "territoriality"],
    "Developing Mind": ["empathy-rudimentary", "joy-response", "grief-response", "social-bonding", "curiosity-advanced", "playfulness", "pattern-recognition"],
    "Proto-Sapient": ["tool-use", "symbolic-thought", "planning-basic", "social-hierarchy", "self-recognition-advanced"],
    "Sapient Being": ["abstract-logic", "long-term-planning", "ethical-framework", "creativity-expressive", "existential-inquiry", "knowledge-seeking"]
}
current_evolutionary_stage_index = 0

# --- Trait Similarity Colors ---
DEFAULT_COLORS_PALETTE = [
    "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#6A5ACD", "#FF69B4", "#00CED1", "#FFA07A", 
    "#9370DB", "#3CB371", "#F08080", "#ADD8E6", "#90EE90", "#FFFFE0", "#C8A2C8", "#DB7093", 
    "#AFEEEE", "#F5DEB3", "#DDA0DD", "#8FBC8F", "#FA8072", "#B0C4DE", "#98FB98", "#FAFAD2", "#E6E6FA" 
]
DEFAULT_TRAIT_COLOR = "#B0B0B0"

# === Dynamic Soul Trait Logic ===
# Adjusted initial traits to fit the "Primordial Essence" or early "Microorganism" idea
initial_traits = {"energy-level": 0.5, "reactivity-to-stimuli": 0.6, "structural-integrity": 0.4}
traits = initial_traits.copy() # Current state of traits
history = []  # List of dictionaries, each a snapshot of traits at a step
conversation_history = []  # Stores all AI utterances: analytical statements, system messages
evolving_theories = [] # Stores AI's generated theories: {text: "...", timestamp: "...", step: ..., stage: "..."}
trait_first_seen = {k: 0 for k in initial_traits}  # Trait name -> step first seen

clients = set() # Set of connected WebSocket clients
user_prompt_override = None  # For overriding the main analytical prompt
user_gemini_api_key = None # Stores validated key from /set_api_keys

# --- Helper Functions (Trait Stats, Correlations, Similarity Colors) ---
def get_trait_statistics(trait_history_matrix: np.ndarray, trait_names: list[str], current_traits_dict: dict, window_size: int = 10) -> dict:
    stats = {}
    if not isinstance(trait_history_matrix, np.ndarray) or trait_history_matrix.ndim != 2 or trait_history_matrix.shape[0] == 0:
        # Fallback if history matrix is not usable
        for name, val in current_traits_dict.items():
            stats[name] = {"current": round(val, 3), "change_last_step": 0.0, 
                           "avg_recent_N": round(val, 3), "trend_slope_recent_N": 0.0, "window_N": 0}
        return stats

    num_history_steps, num_traits_in_hist_matrix = trait_history_matrix.shape

    if num_traits_in_hist_matrix != len(trait_names): # Should not happen if called correctly
        print(f"Warning: Mismatch between trait_names ({len(trait_names)}) and history_matrix columns ({num_traits_in_hist_matrix})")
        # Fallback to current_traits_dict only
        for name, val in current_traits_dict.items():
            stats[name] = {"current": round(val, 3), "change_last_step": 0.0, 
                           "avg_recent_N": round(val, 3), "trend_slope_recent_N": 0.0, "window_N": 0}
        return stats
        
    for i, trait_name in enumerate(trait_names):
        current_value = current_traits_dict.get(trait_name, trait_history_matrix[-1, i] if num_history_steps > 0 else 0.0)
        
        actual_window_size = min(window_size, num_history_steps)
        recent_values = trait_history_matrix[max(0, num_history_steps - actual_window_size):num_history_steps, i]
        
        change = 0.0
        if num_history_steps >= 2: change = trait_history_matrix[-1, i] - trait_history_matrix[-2, i]
        elif num_history_steps == 1: change = trait_history_matrix[-1, i] 

        avg_recent = np.mean(recent_values) if recent_values.size > 0 else current_value
        trend_slope = 0.0
        if recent_values.size >= 2:
            try: trend_slope = np.polyfit(np.arange(len(recent_values)), recent_values, 1)[0]
            except np.linalg.LinAlgError: trend_slope = 0.0
                
        stats[trait_name] = {
            "current": round(current_value, 3), "change_last_step": round(change, 3),
            f"avg_recent_{len(recent_values)}": round(avg_recent, 3), 
            f"trend_slope_recent_{len(recent_values)}": round(trend_slope, 3),
            "window_N": len(recent_values)
        }
    # Add stats for any current traits not in historical trait_names (e.g. brand new)
    for trait_name_curr, current_value_curr in current_traits_dict.items():
        if trait_name_curr not in stats:
             stats[trait_name_curr] = {"current": round(current_value_curr,3), "change_last_step": 0.0, 
                                    "avg_recent_0": round(current_value_curr,3), "trend_slope_recent_0": 0.0, "window_N":0}
    return stats

def get_trait_correlations(trait_history_matrix: np.ndarray, trait_names: list[str], min_corr_steps: int = 5, top_n: int = 3) -> list[str]:
    if (not isinstance(trait_history_matrix, np.ndarray) or trait_history_matrix.ndim != 2 or 
        trait_history_matrix.shape[0] < min_corr_steps or trait_history_matrix.shape[1] < 2 or
        trait_history_matrix.shape[1] != len(trait_names)):
        return ["Not enough data for correlation analysis."]
    try: corr_matrix = np.corrcoef(trait_history_matrix, rowvar=False)
    except Exception as e: print(f"Correlation error: {e}"); return ["Error in correlation calculation."]
    if not isinstance(corr_matrix, np.ndarray) or corr_matrix.ndim != 2 or corr_matrix.shape[0] != len(trait_names):
        return ["Correlation matrix unexpected shape."]

    correlations = []
    for i in range(len(trait_names)):
        for j in range(i + 1, len(trait_names)):
            val = corr_matrix[i, j]
            if not np.isnan(val): correlations.append(((trait_names[i], trait_names[j]), val))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    if not correlations: return ["No calculable correlations."]
    output = [f"'{t1}' & '{t2}' are {('strongly' if abs(v)>0.7 else 'moderately' if abs(v)>0.4 else 'weakly')} {('positively' if v>0 else 'negatively')} correlated ({v:.2f})" 
              for ((t1, t2), v) in correlations[:top_n] if abs(v) > 0.3] # Threshold for reporting
    return output if output else ["No notable correlations found (threshold > 0.3)."]

def calculate_trait_similarities_and_colors(
    trait_names_with_history: list[str], trait_history_matrix: np.ndarray, 
    similarity_threshold: float = 0.95, min_history_steps_for_similarity: int = 3,
    colors_palette: list[str] = None
) -> dict[str, str]:
    if colors_palette is None: colors_palette = DEFAULT_COLORS_PALETTE
    initial_colors = {name: colors_palette[i % len(colors_palette)] for i, name in enumerate(trait_names_with_history)}
    if (not trait_names_with_history or len(trait_names_with_history) < 2 or 
        not isinstance(trait_history_matrix, np.ndarray) or trait_history_matrix.ndim != 2 or 
        trait_history_matrix.shape[1] != len(trait_names_with_history) or 
        trait_history_matrix.shape[0] < min_history_steps_for_similarity):
        return initial_colors
    try: sim_matrix = cosine_similarity(trait_history_matrix.T)
    except Exception as e: print(f"Cosine similarity error: {e}"); traceback.print_exc(); return initial_colors
    assigned_mask = [False] * len(trait_names_with_history); color_idx, grouped_colors = 0, {}
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

# --- Reusable Gemini API Call Function ---
def call_gemini_api(prompt_text: str, model_name: str, context_for_log: str = "Gemini Call", temperature: float = 0.8, top_p: float = 0.95, top_k: int = 40) -> str:
    # ... (This function remains the same as the one provided in the previous response) ...
    api_key_to_use = user_gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key_to_use: print(f"{context_for_log}: CRITICAL - No API key."); return f"Error: No Gemini API key for {context_for_log}."
    try: genai.configure(api_key=api_key_to_use)
    except Exception as e: print(f"{context_for_log}: CRITICAL - API config error: {e}"); return f"Error: Gemini API Key Config Error for {context_for_log}."
    gc = genai.types.GenerationConfig(temperature=temperature,top_p=top_p,top_k=top_k,max_output_tokens=300) # Increased max_output_tokens
    ss = [{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
    # print(f"{context_for_log}: Model '{model_name}' prompt (start): '{prompt_text[:70]}...'") # Keep for debugging if needed
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt_text,generation_config=gc,safety_settings=ss,request_options={"timeout":30})
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            gt = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p,'text')).strip()
            if gt: return gt
        brm, fr = "", "Unknown";
        if hasattr(response,'prompt_feedback') and response.prompt_feedback.block_reason: brm=f" (Block: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason})"
        if hasattr(response,'candidates') and response.candidates and response.candidates[0].finish_reason: fr=response.candidates[0].finish_reason.name
        if fr not in ["STOP","MAX_TOKENS"]: brm+=f" (Finish: {fr})"
        erm=f"Error: Empty/problematic response from '{model_name}' for {context_for_log}{brm}."
        print(f"{context_for_log}: {erm}"); 
        if hasattr(response,'prompt_feedback'): print(f"PF: {response.prompt_feedback}")
        if hasattr(response,'candidates'): print(f"Cand: {response.candidates}")
        return erm
    except google.api_core.exceptions.NotFound as e: erm=f"Error: Model '{model_name}' NOT FOUND for {context_for_log}. {e}"; print(f"{context_for_log}: CRIT - {erm}"); return erm
    except google.api_core.exceptions.InvalidArgument as e: erm=f"Error: INVALID API KEY/Argument for {context_for_log}. {e}"; print(f"{context_for_log}: CRIT - {erm}"); return erm
    except Exception as e: erm=f"Error: Unexpected during {context_for_log} with '{model_name}'. {e}"; print(f"{context_for_log}: ERR - {erm}"); traceback.print_exc(); return erm

# --- Stage-Aware AI Interaction Functions ---
def generate_analytical_statement(current_traits_stats: dict, correlation_summary: list[str], stage_name: str, step: int) -> str:
    # ... (This function remains the same as the one provided in the previous response) ...
    stats_str_parts = []
    for name, data in current_traits_stats.items(): stats_str_parts.append(f"  - '{name}': {data['current']} (change: {data.get('change_last_step','N/A')}, avg_last_{data.get('window_N','N/A')}: {data.get(next(k for k in data if k.startswith('avg_recent_')),'N/A')}, trend_last_{data.get('window_N','N/A')}: {data.get(next(k for k in data if k.startswith('trend_slope_recent_')),'N/A')})")
    stats_summary_str = "\n".join(stats_str_parts); corr_summary_str = "\n".join(f"  - {corr}" for corr in correlation_summary)
    prompt = (f"You are an analytical AI entity at step {step}, in evolutionary stage: '{stage_name}'. Your understanding must be appropriate for this stage. Focus on objective patterns from your internal data.\n\nDATA:\n1. Trait Stats (val, change, avg, trend):\n{stats_summary_str}\n2. Trait Correlations:\n{corr_summary_str}\n\nINSTRUCTIONS (for a '{stage_name}'):\n1. Identify 1-2 salient patterns, anomalies, or changes relevant to your stage.\n2. Formulate a concise observation or question from this data, framed from your current stage's perspective.\n3. Max 2-3 brief sentences.\n\nANALYTICAL STATEMENT:")
    return call_gemini_api(prompt, "models/gemini-2.0-flash", f"Analytical Statement ({stage_name})", temperature=0.6, top_p=0.9)

def generate_updated_theory(current_traits_stats: dict, correlation_summary: list[str], last_analytical_statement: str, previous_theories: list, stage_name: str, step: int) -> str:
    # ... (This function remains the same as the one provided in the previous response) ...
    stats_str_parts = []
    for name, data in current_traits_stats.items(): stats_str_parts.append(f"  - '{name}': {data['current']} (change: {data.get('change_last_step','N/A')}, avg_last_{data.get('window_N','N/A')}: {data.get(next(k for k in data if k.startswith('avg_recent_')),'N/A')}, trend_last_{data.get('window_N','N/A')}: {data.get(next(k for k in data if k.startswith('trend_slope_recent_')),'N/A')})")
    stats_summary_str = "\n".join(stats_str_parts); corr_summary_str = "\n".join(f"  - {corr}" for corr in correlation_summary)
    prev_theories_str = "\n".join([f"  - (Stage: '{t.get('stage', 'Unknown')}', Step {t['step']}) \"{t['text']}\"" for t in previous_theories]) if previous_theories else "  - No prior theories."
    prompt = (f"You are an AI developing theories about your internal functioning. Stage: '{stage_name}', Step: {step}.\n\nDATA ANALYSIS:\n1. Trait Stats:\n{stats_summary_str}\n2. Trait Correlations:\n{corr_summary_str}\n3. Latest Observation (this step):\n  - \"{last_analytical_statement}\"\n\nPREVIOUS THEORIES:\n{prev_theories_str}\n\nINSTRUCTIONS (Feedback Loop for a '{stage_name}'):\n1. Review your *most recent* previous theory considering CURRENT DATA & LATEST OBSERVATION.\n2. Does new info support, contradict, or refine it, given your stage?\n3. Formulate an UPDATED or NEW concise theory (2-4 sentences) appropriate for a '{stage_name}'. Explain reasoning based on data & stage.\n\nTHEORY & REASONING:")
    return call_gemini_api(prompt, "models/gemini-2.0-flash", f"Theory Update ({stage_name})", temperature=0.7, top_p=0.9)

def suggest_trait_evolution_analytically(current_traits_stats: dict, current_theory_text: str|None, current_traits_dict: dict, stage_name: str, stage_trait_concepts: list[str], step: int) -> tuple[str|None, str|None]:
    # ... (This function remains the same as the one provided in the previous response) ...
    stats_str_parts = []
    for name, data in current_traits_stats.items(): stats_str_parts.append(f"  - '{name}': {data['current']} (trend_last_{data.get('window_N','N/A')}: {data.get(next(k for k in data if k.startswith('trend_slope_recent_')),'N/A')})")
    stats_summary_str = "\n".join(stats_str_parts); initial_tn_str = ", ".join(f"'{k}'" for k in initial_traits.keys());
    non_initial_t = [k for k in current_traits_dict if k not in initial_traits]; removable_t_str = ", ".join(f"'{k}'" for k in non_initial_t) if non_initial_t else "none"
    stage_c_str = ", ".join(f"'{c}'" for c in stage_trait_concepts) if stage_trait_concepts else "general concepts"
    prompt = (f"Analytical AI at step {step}, stage '{stage_name}', managing traits.\nStats (val, trend):\n{stats_summary_str}\nTheory: \"{current_theory_text or 'None'}\"\nCore (permanent): {initial_tn_str}\nRemovable: {removable_t_str}\nStage concepts: {stage_c_str}\n\nSuggest (or 'none'):\n1. New single, lowercase, alphanumeric word trait relevant to stage/theory/data.\n2. Existing non-core trait for removal if redundant/irrelevant.\nFormat:\nNew trait: <name_or_none>\nRemove trait: <name_or_none>")
    resp_text = call_gemini_api(prompt, "models/gemini-2.0-flash", f"Trait Suggestion ({stage_name})", temperature=0.65)
    nt_s, rt_s = None, None
    if resp_text and not resp_text.startswith("Error:"):
        for line in resp_text.splitlines():
            ll = line.lower().strip()
            if ll.startswith("new trait:"): c = ll.split(":",1)[1].strip(); nt_s = c if c and c!='none' and len(c.split())==1 and c.isalnum() and not any(uc.isupper() for uc in c) else None
            if ll.startswith("remove trait:"): c = ll.split(":",1)[1].strip(); rt_s = c if c and c!='none' and c in current_traits_dict and c not in initial_traits else None
    return nt_s, rt_s

# --- FastAPI Endpoints & Simulation ---
@app.post("/set_prompt") # Same as before
async def set_prompt_endpoint(request: Request):
    global user_prompt_override; data = await request.json(); new_prompt = data.get("prompt")
    if isinstance(new_prompt, str) and new_prompt.strip(): user_prompt_override = new_prompt; print(f"User prompt override set.")
    elif new_prompt is None or (isinstance(new_prompt, str) and not new_prompt.strip()): user_prompt_override = None; print("User prompt override cleared.")
    else: return JSONResponse({"status":"error","message":"Invalid prompt format"},status_code=400)
    return JSONResponse({"status":"ok","prompt_set_to": "Custom" if user_prompt_override else "Default System Prompt"})

@app.post("/set_api_keys") # Same as before
async def set_api_keys(request: Request):
    global user_gemini_api_key
    data = await request.json(); gemini_key_req = data.get("gemini_api_key"); errors = {}
    print(f"/set_api_keys: Received. Gemini key provided: {'Y' if gemini_key_req else 'N'}")
    if gemini_key_req:
        try:
            print(f"/set_api_keys: Validating Gemini key ...{gemini_key_req[-4:]}")
            genai.configure(api_key=gemini_key_req)
            model_test = "models/gemini-2.0-flash" # User confirmed
            model = genai.GenerativeModel(model_test)
            print(f"/set_api_keys: Test call to '{model_test}'...")
            test_resp = model.generate_content("Test.", request_options={"timeout":10})
            if test_resp.candidates and test_resp.candidates[0].content and test_resp.candidates[0].content.parts and "".join(p.text for p in test_resp.candidates[0].content.parts).strip():
                user_gemini_api_key = gemini_key_req; print(f"/set_api_keys: Key ...{user_gemini_api_key[-4:]} VALIDATED & SET.")
            else:
                br=getattr(getattr(test_resp,'prompt_feedback',None),'block_reason',None); bm=getattr(getattr(test_resp,'prompt_feedback',None),'block_reason_message',None)
                fr_obj=getattr(test_resp.candidates[0] if test_resp.candidates else None,'finish_reason',None); fr=fr_obj.name if fr_obj else "UNK_FIN"
                brm=f" (Block:{bm or br},Finish:{fr})" if br or (fr not in ["STOP","MAX_TOKENS"]) else f" (Finish:{fr})"
                errors["gemini_api_key"]=f"Invalid Key (Test to '{model_test}' empty/blocked{brm})."
                print(f"/set_api_keys: FAIL. {errors['gemini_api_key']}")
        except Exception as e: errors["gemini_api_key"]=f"Key validation error: {type(e).__name__} - {str(e)[:100]}..."; print(f"/set_api_keys: FAIL ex. {errors['gemini_api_key']}"); traceback.print_exc()
    else: errors["gemini_api_key"]="No Gemini key provided."
    if errors: user_gemini_api_key=None; return JSONResponse({"status":"error","errors":errors},status_code=400)
    return JSONResponse({"status":"ok","message":"Gemini API key accepted."})

async def soul_simulation():
    global traits, history, conversation_history, trait_first_seen, evolving_theories, current_evolutionary_stage_index
    step = 0; initial_trait_set = set(initial_traits.keys()); print("Soul simulation loop started.")
    theory_interval, trait_evo_interval, stage_check_interval = 5, 7, 15 # Frequencies
    stats_window, corr_min_steps, corr_top_n = 15, 10, 3 # Analysis params

    while True:
        current_stage_info = EVOLUTIONARY_STAGES[current_evolutionary_stage_index]
        current_stage_name = current_stage_info["name"]
        print(f"\n--- Step {step} | Stage: {current_stage_name} ---")

        for k in traits: traits[k] = round(min(1.0,max(0.0,traits[k]+random.uniform(-0.05,0.05))),3) # Use 3 decimal places
        history.append(traits.copy());
        if len(history)>200: history=history[-200:]
        
        current_traits_snap = traits.copy(); ctna = list(current_traits_snap.keys())
        aptl_pca,pcp,cl,X_pca = [],[],[],np.array([])
        ftc = {n:DEFAULT_COLORS_PALETTE[i%len(DEFAULT_COLORS_PALETTE)] for i,n in enumerate(ctna)}
        mhfa_pca = 3

        if len(history)>=mhfa_pca:
            ats_pca_s={k for s in history for k in s}; aptl_pca=sorted(list(ats_pca_s))
            if aptl_pca:
                Xl_pca=[[s.get(kt,0.0)for kt in aptl_pca]for s in history]; X_pca=np.array(Xl_pca)
                if X_pca.shape[0]>=mhfa_pca and X_pca.shape[1]>0:
                    npc=min(2,X_pca.shape[0],X_pca.shape[1])
                    if npc>=1: try:pca=PCA(n_components=npc);pcp=pca.fit_transform(X_pca).tolist() except:pass
                    nkc=min(4,X_pca.shape[0])
                    if nkc>=1: try:km=KMeans(n_clusters=nkc,n_init='auto',random_state=0);cl=km.fit_predict(X_pca).tolist() except:pass
                    if X_pca.ndim==2 and X_pca.shape[0]>=1 and X_pca.shape[1]==len(aptl_pca):
                        cfs=calculate_trait_similarities_and_colors(aptl_pca,X_pca,min_history_steps_for_similarity=mhfa_pca); ftc.update(cfs)
                    uc=set(ftc.values());ap=[c for c in DEFAULT_COLORS_PALETTE if c not in uc] or DEFAULT_COLORS_PALETTE;cin=0
                    for n in ctna: 
                        if n not in ftc: ftc[n]=ap[cin%len(ap)];cin+=1
                    ftc={n:ftc.get(n,DEFAULT_TRAIT_COLOR) for n in ctna}
        
        trait_stats={}, corr_sum_list=["Not enough data."]
        hist_for_analysis=history[max(0,len(history)-stats_window):]
        min_steps_for_stats=3
        if len(hist_for_analysis)>=min_steps_for_stats:
            traits_in_win_set={k for s in hist_for_analysis for k in s}; traits_in_win_list=sorted(list(traits_in_win_set))
            if traits_in_win_list:
                X_an_list=[[s.get(kt,0.0)for kt in traits_in_win_list]for s in hist_for_analysis]; X_an_matrix=np.array(X_an_list)
                if X_an_matrix.ndim==2 and X_an_matrix.shape[0]>0 and X_an_matrix.shape[1]>0:
                    trait_stats=get_trait_statistics(X_an_matrix,traits_in_win_list,current_traits_snap,len(hist_for_analysis))
                    corr_sum_list=get_trait_correlations(X_an_matrix,traits_in_win_list,max(3,corr_min_steps),corr_top_n)

        analytical_statement = generate_analytical_statement(trait_stats,corr_sum_list,current_stage_name,step)
        conversation_history.append({"speaker":f"AI Soul ({current_stage_name})","text":analytical_statement,"timestamp":datetime.now(timezone.utc).isoformat(),"type":"analysis", "step": step})
        print(f"S{step} Analytical Statement: '{analytical_statement[:70]}...'")

        if step>0 and step%theory_interval==0:
            print(f"S{step} Updating theory (Stage: {current_stage_name})...");
            new_theory=generate_updated_theory(trait_stats,corr_sum_list,analytical_statement,evolving_theories[-1:],current_stage_name,step)
            if new_theory and not new_theory.startswith("Error:"):
                evolving_theories.append({"text":new_theory,"timestamp":datetime.now(timezone.utc).isoformat(),"step":step,"stage":current_stage_name})
                print(f"S{step} Theory: '{new_theory[:70]}...'");
                if len(evolving_theories)>10:evolving_theories=evolving_theories[-10:]
        
        if step>0 and step%trait_evo_interval==0:
            print(f"S{step} Suggesting trait evolution (Stage: {current_stage_name})...");
            theory_ctx=evolving_theories[-1]['text'] if evolving_theories else "No specific theory yet."
            stage_ctx_concepts=STAGE_RELEVANT_TRAIT_CONCEPTS.get(current_stage_name,[])
            new_sugg,rem_sugg=suggest_trait_evolution_analytically(trait_stats,theory_ctx,current_traits_snap,current_stage_name,stage_ctx_concepts,step)
            if new_sugg and new_sugg not in traits:traits[new_sugg]=round(random.uniform(0.2,0.6),3);trait_first_seen[new_sugg]=step;print(f"S{step} AI NEW trait '{new_sugg}'.")
            if rem_sugg and rem_sugg in traits and rem_sugg not in initial_trait_set:del traits[rem_sugg];trait_first_seen.pop(rem_sugg,None);print(f"S{step} AI REMOVAL of trait '{rem_sugg}'.")

        if step>0 and step%stage_check_interval==0 and current_evolutionary_stage_index<len(EVOLUTIONARY_STAGES)-1:
            adv=False; csi = current_evolutionary_stage_index # Store before potential change
            # --- More sophisticated advancement criteria needed here based on traits, theories, or complexity ---
            if EVOLUTIONARY_STAGES[csi]["name"] == "Primordial Essence" and step > (10 + csi*5) : adv = True 
            elif EVOLUTIONARY_STAGES[csi]["name"] == "Microorganism" and traits.get('motility',0) > 0.6 and step > (25 + csi*5) : adv = True
            elif EVOLUTIONARY_STAGES[csi]["name"] == "Colonial Organism" and traits.get('group-synchrony',0) > 0.5 and len(traits) > (len(initial_traits)+2) and step > (40 + csi*5): adv = True
            elif EVOLUTIONARY_STAGES[csi]["name"] == "Instinctual Creature" and (traits.get('basic-memory',0) > 0.5 or traits.get('social-bonding',0) > 0.3) and step > (60 + csi*10): adv = True
            elif EVOLUTIONARY_STAGES[csi]["name"] == "Developing Mind" and traits.get('curiosity-advanced',0) > 0.6 and traits.get('pattern-recognition',0) > 0.5 and len(evolving_theories) > 1 and step > (80 + csi*10): adv = True
            elif EVOLUTIONARY_STAGES[csi]["name"] == "Proto-Sapient" and traits.get('abstract-logic',0) > 0.4 and traits.get('planning-basic',0) > 0.4 and len(evolving_theories) > 3 and step > (100 + csi*10): adv = True


            if adv:
                old_stage_name = EVOLUTIONARY_STAGES[csi]["name"]
                current_evolutionary_stage_index+=1
                new_stage_name=EVOLUTIONARY_STAGES[current_evolutionary_stage_index]["name"]
                evo_msg=f"EVOLUTIONARY ADVANCEMENT: Stage {csi} '{old_stage_name}' --> Stage {current_evolutionary_stage_index} '{new_stage_name}' at step {step}."
                print(evo_msg);conversation_history.append({"speaker":"SYSTEM (Evolution)","text":evo_msg,"timestamp":datetime.now(timezone.utc).isoformat(),"type":"evolution", "step": step})
                # Add 1-2 new traits relevant to the new stage
                new_stage_concepts = STAGE_RELEVANT_TRAIT_CONCEPTS.get(new_stage_name, [])
                added_count = 0
                for concept in new_stage_concepts:
                    if concept not in traits and added_count < 2:
                        traits[concept] = round(random.uniform(0.3,0.7),3); trait_first_seen[concept]=step
                        print(f"S{step}: Trait '{concept}' emerged with stage '{new_stage_name}'."); added_count+=1
                    if added_count >= 2: break
        
        if len(conversation_history)>30:conversation_history=conversation_history[-30:]
        data_to_send={"traits":current_traits_snap,"trait_names":ctna,"history":history,"step":step,
                      "pca_points":pcp,"clusters":cl,"all_traits_pca_order":aptl_pca,
                      "conversation_history":conversation_history, "trait_first_seen":trait_first_seen,
                      "trait_colors":ftc, "evolving_theories":evolving_theories,
                      "current_evolutionary_stage":EVOLUTIONARY_STAGES[current_evolutionary_stage_index]["name"],
                      "current_stage_description":EVOLUTIONARY_STAGES[current_evolutionary_stage_index]["description"]}
        for wc in list(clients):
            try:await wc.send_json(data_to_send)
            except:clients.discard(wc)
        step+=1; await asyncio.sleep(15)

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
# Ensure your updated index.html (provided separately) is in the static/index.html path
app.mount("/",StaticFiles(directory=static_dir_name,html=True),name="static")

if __name__=="__main__":
    import uvicorn;mn="chatbot";p=int(os.getenv("PORT",10000))
    print(f"🚀 Starting Uvicorn server on http://0.0.0.0:{p}")
    print(f"👉 Running FastAPI 'app' from '{mn}.py'")
    print("🔑 Ensure GEMINI_API_KEY (environment variable) is set if not using /set_api_keys, or if submitted key is invalid.")
    print(f"🧬 Soul starts at stage: {EVOLUTIONARY_STAGES[current_evolutionary_stage_index]['name']}")
    uvicorn.run(f"{mn}:app",host="0.0.0.0",port=p,reload=True)