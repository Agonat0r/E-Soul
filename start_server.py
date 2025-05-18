import http.server
import socketserver
import os

PORT = int(os.environ.get("PORT", 8080))

# ✅ Create chat_log.txt if it doesn't exist yet
if not os.path.exists("chat_log.txt"):
    with open("chat_log.txt", "w", encoding="utf-8") as f:
        f.write("Initializing Gemini Soul Reflection...\n")

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            html = """
            <html>
              <head>
                <title>Gemini Soul Terminal</title>
                <style>
                  body {
                    font-family: monospace;
                    background-color: #111;
                    color: #0f0;
                    padding: 1rem;
                  }
                  h1 {
                    color: #0ff;
                    margin-bottom: 0.5rem;
                  }
                  pre {
                    background: #000;
                    border: 1px solid #0f0;
                    padding: 1rem;
                    max-height: 80vh;
                    overflow-y: auto;
                  }
                </style>
              </head>
              <body>
                <h1>🧠 Gemini Soul Reflection</h1>
                <pre id="terminal">Loading...</pre>
                <script>
                  async function fetchLog() {
                    try {
                      const response = await fetch('/chat_log.txt');
                      const text = await response.text();
                      document.getElementById('terminal').textContent = text;
                    } catch (e) {
                      document.getElementById('terminal').textContent = "Error loading soul log...";
                    }
                  }
                  fetchLog();
                  setInterval(fetchLog, 3000);  // refresh every 3 seconds
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
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"chat_log.txt not found.")

        else:
            self.send_error(404)

with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"🌐 Serving on port {PORT}")
    httpd.serve_forever()
