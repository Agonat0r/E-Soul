import time
import os
import google.generativeai as genai

print("🚀 Starting Gemini 2.0 Flash Self-Chat...")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY is missing.")
    exit(1)

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    # Initial message
    last_reply = "Hello, who are you?"
    history = []

    while True:
        print("Prompting Gemini with:", last_reply)

        response = model.generate_content(last_reply)
        reply = response.text.strip()
        print("🤖 Gemini:", reply)

        # Add to history
        history.append(f"> User: {last_reply}")
        history.append(f"> Gemini: {reply}")

        # Save history to log
        with open("chat_log.txt", "w", encoding="utf-8") as f:
            f.write("Gemini 2.0 Self-Chat:\n\n")
            f.write("\n".join(history[-20:]))  # Show last 10 exchanges

        # Feed Gemini's own reply back into itself
        last_reply = reply
        time.sleep(10)

except Exception as e:
    print("❌ Error:", e)
    time.sleep(10)
