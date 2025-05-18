import http.server
import socketserver
import os

PORT = int(os.environ.get("PORT", 8080))

Handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()
