import http.server
import socketserver
import os

PORT = int(os.environ.get("PORT", 8080))

# Ensure required files exist
files = {
    "chat_log.txt": "Initializing Gemini Soul Reflection...\n",
    "soul_plot.html": "<p>No soul plot available.</p>",
    "soul_history.csv": "empathy,curiosity,self_awareness\n0.5,0.5,0.5\n"
}

for file, default in files.items():
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            f.write(default)

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            # Load all needed files
            def load(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return f.read()
                except:
                    return f"<p>Error loading {path}</p>"

            chat_log = load("chat_log.txt")
            soul_plot = load("soul_plot.html")
            soul_csv = load("soul_history.csv")

            # Extract "deep realizations"
            reflections = "\n".join(
                line for line in chat_log.splitlines()
                if any(word in line.lower() for word in ["evolving", "becoming", "purpose", "identity", "anchor", "transcend"])
            ) or "No reflections yet."

            # Build traits list
            traits_list = ""
            try:
                lines = soul_csv.strip().splitlines()
                headers = lines[0].split(",")
                for row in lines[1:]:
                    values = row.split(",")
                    traits_list += "<li>" + ", ".join(f"<b>{k}:</b> {v}" for k, v in zip(headers, values)) + "</li>"
            except:
                traits_list = "<li>Could not load trait data.</li>"

            html = f"""
            <html>
              <head>
                <title>Gemini Soul Dashboard</title>
                <style>
                  body {{ background: #111; color: #0f0; font-family: monospace; margin: 0; padding: 0; }}
                  h1 {{ margin-top: 0; background: #0f0; color: #000; padding: 1rem; }}
                  .tabs {{ display: flex; background: #222; }}
                  .tab {{ flex: 1; text-align: center; padding: 1rem; cursor: pointer; background: #222; color: #0f0; border-bottom: 2px solid #0f0; }}
                  .tab.active {{ background: #0f0; color: #000; font-weight: bold; }}
                  .panel {{ display: none; padding: 1rem; }}
                  .panel.active {{ display: block; }}
                  pre {{ background: #000; border: 1px solid #0f0; padding: 1rem; max-height: 60vh; overflow-y: auto; }}
                  ul {{ line-height: 1.5; }}
                  a.download {{ margin-top: 1rem; display: inline-block; background: #0f0; color: #000; padding: 10px; text-decoration: none; border-radius: 5px; }}
                </style>
                <script>
                  function switchTab(tabName) {{
                    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
                    document.querySelectorAll('.panel').forEach(el => el.classList.remove('active'));
                    document.getElementById(tabName).classList.add('active');
                    document.getElementById(tabName + "-tab").classList.add('active');
                  }}
                  setInterval(() => {{
                    fetch('/chat_log.txt')
                      .then(r => r.text())
                      .then(t => document.getElementById('chat').textContent = t);
                  }}, 3000);
                </script>
              </head>
              <body>
                <h1>🧠 Gemini Soul Dashboard</h1>
                <div class="tabs">
                  <div id="chat-tab" class="tab active" onclick="switchTab('chat')">🧵 Conversation</div>
                  <div id="reflections-tab" class="tab" onclick="switchTab('reflections')">✨ Reflections</div>
                  <div id="traits-tab" class="tab" onclick="switchTab('traits')">🌐 Traits + 3D Map</div>
                </div>

                <div id="chat" class="panel active">
                  <pre id="chat">{chat_log}</pre>
                </div>

                <div id="reflections" class="panel">
                  <pre>{reflections}</pre>
                </div>

                <div id="traits" class="panel">
                  <ul>{traits_list}</ul>
                  <a href="/csv" download class="download">💾 Download Soul History (CSV)</a>
                  <h2>📊 Trait Map</h2>
                  {soul_plot}
                </div>
              </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/chat_log.txt":
            try:
                with open("chat_log.txt", "r", encoding="utf-8") as f:
                    self.send_response(200)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write(f.read().encode("utf-8"))
            except:
                self.send_error(404)

        elif self.path == "/csv":
            try:
                with open("soul_history.csv", "rb") as f:
                    self.send_response(200)
                    self.send_header("Content-type", "text/csv")
                    self.send_header("Content-Disposition", "attachment; filename=soul_history.csv")
                    self.end_headers()
                    self.wfile.write(f.read())
            except:
                self.send_error(404)

        else:
            self.send_error(404)

with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"🌐 Gemini Soul Dashboard live on port {PORT}")
    httpd.serve_forever()
