"""
Fragility Benchmark — localhost dashboard server.
Run:  python server.py
Open: http://localhost:8099
"""

import http.server
import json
import os
from pathlib import Path

PORT = 8099
BASE = Path(__file__).resolve().parent.parent  # experiments/fragility-map

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(BASE / "site"), **kw)

    def do_GET(self):
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            judged = json.loads((BASE / "results" / "judged_responses.json").read_text())
            battery = json.loads((BASE / "prompts" / "battery.json").read_text())

            payload = json.dumps({
                "judged": judged,
                "battery": battery,
            })
            self.wfile.write(payload.encode())
        else:
            super().do_GET()

if __name__ == "__main__":
    print(f"Fragility Benchmark dashboard → http://localhost:{PORT}")
    with http.server.HTTPServer(("", PORT), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
