"""
RetroFit Web Dashboard Launcher
Starts a local web server at http://localhost:8080 and opens the dashboard in your browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run():
    print("=" * 60)
    print(f"🚀 RETROFIT LIVE DASHBOARD RUNNING AT http://localhost:{PORT}")
    print("=" * 60)
    print("Connected to free EMQX MQTT Broker (broker.emqx.io)...")
    
    webbrowser.open(f"http://localhost:{PORT}")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run()
