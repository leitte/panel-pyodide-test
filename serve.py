#!/usr/bin/env python3
import http.server, socketserver, mimetypes, os, re

ROOT = os.path.abspath(".")
PORT = 8000

# Ensure .wasm uses streaming-friendly MIME
mimetypes.add_type("application/wasm", ".wasm")

PYODIDE_DIR = re.compile(r"/static/pyodide-\d+\.\d+\.\d+/")  # adjust if needed

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Long cache for versioned Pyodide assets
        if PYODIDE_DIR.search(self.path):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        # Make sure .wasm has the right content-type (older Pythons can be flaky)
        ctype = self.guess_type(self.path)
        if self.path.endswith(".wasm") and ctype != "application/wasm":
            self.send_header("Content-Type", "application/wasm")
        super().end_headers()

    # Optional: set CORS so you can test cross-origin fetches
    # def send_head(self):
    #     self.send_header("Access-Control-Allow-Origin", "*")
    #     return super().send_head()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving {ROOT} at http://localhost:{PORT}")
        httpd.serve_forever()
