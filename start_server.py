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

            try:
                with open("chat_log.txt", "r", encoding="utf-8") as f:
                    message = f.read()
            except FileNotFoundError:
                message = "No messages yet."

            html = f"<html><body><h1>Gemini Self-Chat</h1><pre>{message}</pre></body></html>"
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_error(404)

with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"Serving on port {PORT}")
    httpd.serve_forever()
