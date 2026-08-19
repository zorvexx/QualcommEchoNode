import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from src.registration.manager import MachineRegistrationManager

class RetroFitUIServer(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == '/api/fleet':
            mgr = MachineRegistrationManager()
            machines = mgr.list_registered_machines()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(machines).encode('utf-8'))
        elif self.path == '/api/telemetry':
            # Load latest computed telemetry state
            latest_path = "data/models/latest_telemetry.json"
            if os.path.exists(latest_path):
                with open(latest_path, 'rb') as f:
                    content = f.read()
            else:
                content = json.dumps({
                    'machine_id': 'LAPTOP_IDLE_01',
                    'similarity': 100.0,
                    'behavior_drift': 0.0,
                    'status': 'KNOWN_NORMAL_STATE',
                    'confidence': 99.0,
                    'state': 0
                }).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(content)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/register':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            mgr = MachineRegistrationManager()
            cfg = mgr.register_machine(
                machine_id=data.get('machine_id', 'DEV_01'),
                machine_name=data.get('machine_name', 'Default Machine'),
                machine_type=data.get('machine_type', 'General'),
                location=data.get('location', 'Facility 1'),
                operator_phone=data.get('operator_phone', '')
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(cfg).encode('utf-8'))

def start_dashboard_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RetroFitUIServer)
    print(f"[UI SERVER] RetroFit Dashboard Server running at http://localhost:{port}/")
    return httpd

if __name__ == "__main__":
    httpd = start_dashboard_server(8080)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[UI SERVER] Dashboard server stopped.")
