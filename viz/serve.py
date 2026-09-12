"""Local HTTPS server for the WebXR viewer.

Generates a self-signed certificate on first run so WebXR AR works
over Wi-Fi from a phone without USB debugging.

Usage:
  python -m viz.serve --port 8080 --scene-dir viz/output/subject001

On your phone, open https://<your-ip>:8080 and accept the certificate warning.
"""

import argparse
import functools
import http.server
import os
import socketserver
import ssl
import subprocess
import sys


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

    with socketserver.TCPServer(('', args.port), handler) as httpd:
        if not args.no_ssl:
            try:
                cert_file, key_file = ensure_cert(cert_dir)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert_file, key_file)
                httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
            except Exception as e:
                print(f'SSL setup failed ({e}), falling back to HTTP')
                print('Install OpenSSL or use --no-ssl flag')
                protocol = 'http'

        print(f'Desktop: {protocol}://localhost:{args.port}')
        print(f'Mobile:  {protocol}://{ip}:{args.port}')
        if protocol == 'https':
            print(f'Accept the certificate warning on your phone to continue.')
        print('Press Ctrl+C to stop')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nStopped.')


if __name__ == '__main__':
    main()
