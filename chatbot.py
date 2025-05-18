import time
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

message = "Hello, who are you?"

while True:
    try:
        response = chat.send_message(message)
        message = response.text.strip()

        with open("chat_log.txt", "w", encoding="utf-8") as f:
            f.write(message)

        print("Gemini:", message)
        time.sleep(10)
    except Exception as e:
        print("Error:", e)
        time.sleep(5)
