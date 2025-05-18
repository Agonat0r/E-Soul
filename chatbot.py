import time
import google.generativeai as genai
import os

print("Starting Gemini chatbot...")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY is missing.")
    exit(1)

genai.configure(api_key=api_key)

try:
    # Use the supported model for chat-style interaction
    model = genai.GenerativeModel("models/chat-bison-001")
    chat = model.start_chat(history=[])

    message = "Hello, who are you?"

    while True:
        response = chat.send_message(message)
        message = response.text.strip()

        print("🤖 Gemini:", message)

        with open("chat_log.txt", "w", encoding="utf-8") as f:
            f.write(message)

        time.sleep(10)

except Exception as e:
    print("❌ Error during Gemini loop:", e)
    time.sleep(10)
