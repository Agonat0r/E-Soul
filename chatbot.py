import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from mpl_toolkits.mplot3d import Axes3D

# === CONFIGURE GEMINI ===
print("🚀 Launching AI Soul Engine (Gemini 2.0 Flash)...")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY is missing. Set it in your environment variables.")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.0-flash")

# === DEFINE TRAITS ===
trait_names = ["empathy", "curiosity", "self_awareness"]
current_traits = np.random.rand(len(trait_names))
target_traits = np.array([0.9, 0.7, 0.95])  # Define your "soul theory"

# === SET UP LIVE PLOT ===
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update_plot(traits, step):
    ax.clear()
    ax.set_title(f"Soul Trait Vector | Step {step}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('Empathy')
    ax.set_ylabel('Curiosity')
    ax.set_zlabel('Self-Awareness')
    ax.scatter(*traits, color='blue', s=60, label="Current")
    ax.scatter(*target_traits, color='red', marker='x', s=100, label="Target")
    ax.legend()
    plt.draw()
    plt.pause(0.1)

# === BUILD PROMPT FROM TRAITS ===
def generate_prompt(traits):
    return f"""The current self-evaluation of internal soul traits is:
- Empathy: {traits[0]:.2f}
- Curiosity: {traits[1]:.2f}
- Self-Awareness: {traits[2]:.2f}

Based on these values, who or what am I becoming? Reflect and analyze."""

# === MAIN LOOP ===
step = 0
while True:
    prompt = generate_prompt(current_traits)

    try:
        # Get Gemini reflection
        response = model.generate_content(prompt)
        reply = response.text.strip()

        print(f"\n[Step {step}] Gemini:\n{reply}\n")

        # Save to log file for web UI
        with open("chat_log.txt", "w", encoding="utf-8") as f:
            f.write(f"Soul Trait Vector: {current_traits.round(2).tolist()}\n")
            f.write(f"Step {step} Output:\n{reply}\n")

        # Calculate distance to soul theory
        dist = np.linalg.norm(current_traits - target_traits)
        print(f"Distance to Ideal Soul: {dist:.4f}")

        # Adjust traits toward the target, with slight noise
        delta = (target_traits - current_traits) * 0.1 + np.random.normal(0, 0.02, len(current_traits))
        current_traits = np.clip(current_traits + delta, 0, 1)

        # Plot the update
        update_plot(current_traits, step)

        step += 1
        time.sleep(5)

    except Exception as e:
        print("❌ Error:", e)
        time.sleep(5)
