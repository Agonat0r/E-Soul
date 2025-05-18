import http.server
import socketserver
import os

PORT = int(os.environ.get("PORT", 8080))

# Ensure required files exist
if not os.path.exists("chat_log.txt"):
    with open("chat_log.txt", "w", encoding="utf-8") as f:
        f.write("Initializing Gemini Soul Reflection...\n")

if not os.path.exists("soul_plot.html"):
    with open("soul_plot.html", "w", encoding="utf-8") as f:
        f.write("<html><body><h1>Soul plot not generated yet.</h1></body></html>")

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            # === Load soul_plot.html into the page ===
            try:
                with open("soul_plot.html", "r", encoding="utf-8") as f:
                    plot_html = f.read()
            except:
                plot_html = "<p>Plot not available.</p>"

            html = f"""
            <html>
              <head>
                <title>Gemini Soul Interface</title>
                <style>
                  body {{
                    font-family: monospace;
                    background-color: #111;
                    color: #0f0;
                    padding: 1rem;
                  }}
                  h1 {{
                    color: #0ff;
                  }}
                  pre {{
                    background: #000;
                    border: 1px solid #0f0;
                    padding: 1rem;
                    max-height: 40vh;
                    overflow-y: auto;
                  }}
                  .download-btn {{
                    background: #0f0;
                    color: #000;
                    padding: 10px 20px;
                    text-decoration: none;
                    font-family: monospace;
                    border-radius: 5px;
                    display: inline-block;
                    margin-top: 20px;
                  }}
                  iframe {{
                    width: 100%;
                    height: 500px;
                    border: 1px solid #0f0;
                    margin-top: 20px;
                    background: #fff;
                  }}
                </style>
              </head>
              <body>
                <h1>🧠 Gemini Soul Reflection</h1>
                <pre id="terminal">Loading...</pre>

                <a href="/csv" download class="download-btn">💾 Download Soul Data (CSV)</a>

                <h2>📊 Soul Trait Evolution (3D)</h2>
                {plot_html}

                <script>
                  async function fetchLog() {{
                    try {{
                      const response = await fetch('/chat_log.txt');
                      const text = await response.text();
                      document.getElementById('terminal').textContent = text;
                    }} catch (e) {{
                      document.getElementById('terminal').textContent = "Error loading log.";
                    }}
                  }}
                  fetchLog();
                  setInterval(fetchLog, 3000);
                </script>
              </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/chat_log.txt":
            try:
                with open("chat_log.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            except:
                self.send_error(404)

        elif self.path == "/csv":
            try:
                with open("soul_history.csv", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=soul_history.csv")
                self.end_headers()
                self.wfile.write(content)
            except:
                self.send_error(404)

        else:
            self.send_error(404)

with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"🌐 Serving full interface on port {PORT}")
    httpd.serve_forever()
