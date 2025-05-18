import http.server
import socketserver
import os

PORT = int(os.environ.get("PORT", 8080))

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            html = """
            <html>
              <head>
                <title>Gemini Self-Chat</title>
                <style>
                  body {
                    font-family: monospace;
                    background-color: #111;
                    color: #0f0;
                    padding: 1rem;
                  }
                  h1 {
                    color: #0ff;
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
                <h1>Gemini Self-Chat</h1>
                <pre id="terminal">Loading...</pre>
                <script>
                  async function fetchLog() {
                    const response = await fetch('/chat_log.txt');
                    const text = await response.text();
                    document.getElementById('terminal').textContent = text;
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
    print(f"Serving on port {PORT}")
    httpd.serve_forever()
