"""Local HTTPS server for the WebXR viewer.

Generates a self-signed certificate on first run so WebXR AR works
over Wi-Fi from a phone without USB debugging.

Includes a WebSocket server on port+1 for phone->laptop state sync
(companion display).

Usage:
  python -m viz.serve --port 8080 --scene-dir viz/output/subject001

On your phone, open https://<your-ip>:8080 and accept the certificate warning.
Companion display: https://<your-ip>:8080?mode=companion
"""

import argparse
import asyncio
import functools
import http.server
import json
import os
import socketserver
import ssl
import subprocess
import sys
import threading


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, viewer_dir=None, scene_dir=None, **kwargs):
        self.viewer_dir = viewer_dir
        self.scene_dir = scene_dir
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        path = path.split('?')[0].split('#')[0]
        path = path.lstrip('/')

        if path.startswith('data/'):
            rel = path[len('data/'):]
            return os.path.join(self.scene_dir, rel)

        scene_path = os.path.join(self.scene_dir, path)
        if path and os.path.exists(scene_path):
            return scene_path

        return os.path.join(self.viewer_dir, path)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        pass  # quiet

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.glb': 'model/gltf-binary',
        '.gltf': 'model/gltf+json',
        '.json': 'application/json',
    }


# ─── WebSocket sync server ───────────────────────────────────────────────────

_ws_clients = set()


async def _ws_handler(websocket):
    """Handle one WebSocket client. Broadcast messages to all others."""
    _ws_clients.add(websocket)
    try:
        async for message in websocket:
            # Forward to every other connected client
            peers = [c for c in _ws_clients if c is not websocket]
            for peer in peers:
                try:
                    await peer.send(message)
                except Exception:
                    pass
    finally:
        _ws_clients.discard(websocket)


def _run_ws_server(port, ssl_context=None):
    """Run the WebSocket server in its own asyncio event loop."""
    import websockets

    async def serve():
        kwargs = {}
        if ssl_context is not None:
            kwargs["ssl"] = ssl_context
        async with websockets.serve(_ws_handler, "", port, **kwargs):
            await asyncio.Future()  # run forever

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(serve())


def start_ws_server(ws_port, ssl_context=None):
    """Launch the WebSocket server in a daemon thread."""
    t = threading.Thread(target=_run_ws_server, args=(ws_port, ssl_context),
                         daemon=True)
    t.start()
    return t


# ─── Utilities ────────────────────────────────────────────────────────────────

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def ensure_cert(cert_dir):
    """Generate a self-signed cert if one doesn't exist yet."""
    cert_file = os.path.join(cert_dir, 'cert.pem')
    key_file = os.path.join(cert_dir, 'key.pem')

    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file

    os.makedirs(cert_dir, exist_ok=True)

    ip = get_local_ip()
    print(f'Generating self-signed certificate for {ip}...')

    subprocess.run([
        'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
        '-keyout', key_file, '-out', cert_file,
        '-days', '365', '-nodes',
        '-subj', f'/CN={ip}',
        '-addext', f'subjectAltName=IP:{ip},IP:127.0.0.1',
    ], check=True, capture_output=True)

    print(f'Certificate saved to {cert_dir}/')
    return cert_file, key_file


def main():
    parser = argparse.ArgumentParser(description='Serve the WebXR aorta viewer over HTTPS')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--scene-dir', default='viz/output',
                        help='Directory containing exported scene')
    parser.add_argument('--no-ssl', action='store_true',
                        help='Use plain HTTP instead of HTTPS')
    parser.add_argument('--no-ws', action='store_true',
                        help='Disable WebSocket sync server')
    args = parser.parse_args()

    viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viewer')
    scene_dir = os.path.abspath(args.scene_dir)
    cert_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.certs')

    handler = functools.partial(
        ViewerHandler,
        viewer_dir=viewer_dir,
        scene_dir=scene_dir,
    )

    ip = get_local_ip()
    protocol = 'http' if args.no_ssl else 'https'
    ws_port = args.port + 1
    ssl_context = None

    with socketserver.TCPServer(('', args.port), handler) as httpd:
        if not args.no_ssl:
            try:
                cert_file, key_file = ensure_cert(cert_dir)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert_file, key_file)
                httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
                ssl_context = ctx
            except Exception as e:
                print(f'SSL setup failed ({e}), falling back to HTTP')
                print('Install OpenSSL or use --no-ssl flag')
                protocol = 'http'

        # Start WebSocket sync server
        ws_protocol = 'wss' if protocol == 'https' else 'ws'
        if not args.no_ws:
            try:
                start_ws_server(ws_port, ssl_context=ssl_context)
                print(f'WebSocket: {ws_protocol}://{ip}:{ws_port}')
            except Exception as e:
                print(f'WebSocket server failed ({e}), sync disabled')
                print('Install websockets: pip install websockets')

        print(f'Desktop:   {protocol}://localhost:{args.port}')
        print(f'Mobile:    {protocol}://{ip}:{args.port}')
        print(f'Companion: {protocol}://{ip}:{args.port}?mode=companion')
        if protocol == 'https':
            print(f'Accept the certificate warning on your phone to continue.')
        print('Press Ctrl+C to stop')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nStopped.')


if __name__ == '__main__':
    main()
