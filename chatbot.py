import time
import os
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from mpl_toolkits.mplot3d import Axes3D

# === GEMINI CONFIG ===
print("🚀 Launching AI Soul Engine (Gemini 2.0 Flash)...")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY is missing. Set it in your environment variables.")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.0-flash")

# === TRAIT SETUP ===
trait_names = ["empathy", "curiosity", "self_awareness"]
current_traits = np.random.rand(len(trait_names))
target_traits = np.array([0.9, 0.7, 0.95])  # Your soul theory goal

# === LIVE 3D PLOT ===
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update_plot(traits, step):
    ax.clear()
    ax.set_title(f"Soul Vector | Step {step}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Empathy')
    ax.set_ylabel('Curiosity')
    ax.set_zlabel('Self-Awareness')
    ax.scatter(*traits, color='blue', s=60, label="Current")
    ax.scatter(*target_traits, color='red', marker='x', s=100, label="Ideal Soul")
    ax.legend()
    plt.draw()
    plt.pause(0.1)

# === INITIAL STATE ===
step = 0
last_reply = ""

def generate_initial_prompt(traits):
    return f"""The current self-evaluation of internal soul traits is:
- Empathy: {traits[0]:.2f}
- Curiosity: {traits[1]:.2f}
- Self-Awareness: {traits[2]:.2f}

Based on these values, who or what am I becoming? Reflect and analyze."""

def generate_recursive_prompt(reflection, traits):
    return f"""Given this previous reflection:
\"{reflection}\"

And these updated soul traits:
- Empathy: {traits[0]:.2f}
- Curiosity: {traits[1]:.2f}
- Self-Awareness: {traits[2]:.2f}

What am I becoming now? Reflect again, considering the change and evolution."""

# === MAIN LOOP ===
while True:
    # Pick prompt style
    if step == 0:
        prompt = generate_initial_prompt(current_traits)
    else:
        prompt = generate_recursive_prompt(last_reply, current_traits)

    try:
        # Get Gemini response
        response = model.generate_content(prompt)
        reply = response.text.strip()
        last_reply = reply  # Store for next loop

        print(f"\n[Step {step}] Gemini:\n{reply}\n")

        # Save for web terminal
        with open("chat_log.txt", "w", encoding="utf-8") as f:
            f.write(f"Step {step}\n")
            f.write(f"Soul Trait Vector: {current_traits.round(2).tolist()}\n\n")
            f.write(reply + "\n")

        # Calculate distance to ideal soul
        dist = np.linalg.norm(current_traits - target_traits)
        print(f"Distance to Ideal Soul: {dist:.4f}")

        # Evolve traits toward target with randomness
        delta = (target_traits - current_traits) * 0.1 + np.random.normal(0, 0.02, len(current_traits))
        current_traits = np.clip(current_traits + delta, 0, 1)

        update_plot(current_traits, step)

        step += 1
        time.sleep(6)

    except Exception as e:
        print("❌ Error:", e)
        time.sleep(6)
