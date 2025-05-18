import time
import os
import google.generativeai as genai

print("🚀 Starting Gemini 2.0 Flash chatbot...")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY is missing.")
    exit(1)

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    # Start the conversation
    prompt_history = ["Hello, who are you?"]

    while True:
        prompt = "\n".join(prompt_history[-5:])  # Limit to last 5 lines for context

        response = model.generate_content(prompt)
        reply = response.text.strip()

        print("🤖 Gemini:", reply)

        # Save to file
        with open("chat_log.txt", "w", encoding="utf-8") as f:
            f.write("Gemini 2.0 Self-Chat:\n")
            for i, line in enumerate(prompt_history):
                f.write(f"> User: {line}\n" if i % 2 == 0 else f"> Gemini: {line}\n")

            f.write(f"> Gemini: {reply}\n")

        # Add reply back into the history
        prompt_history.append(reply)
        time.sleep(10)

except Exception as e:
    print("❌ Error during Gemini loop:", e)
    time.sleep(10)
