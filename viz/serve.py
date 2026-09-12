"""Local development server for the WebXR viewer.

Usage:
  python -m viz.serve --port 8080 --scene-dir viz/output/subject001

Then open: http://localhost:8080
On Android (same Wi-Fi): http://<your-ip>:8080
"""

import argparse
import functools
import http.server
import os
import socketserver


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Serves viewer static files + scene assets from a combined root."""

    def __init__(self, *args, viewer_dir=None, scene_dir=None, **kwargs):
        self.viewer_dir = viewer_dir
        self.scene_dir = scene_dir
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        path = path.split('?')[0].split('#')[0]
        path = path.lstrip('/')

        # /data/* serves from the scene directory
        if path.startswith('data/'):
            rel = path[len('data/'):]
            return os.path.join(self.scene_dir, rel)

        # Check scene dir first for scene.json / aorta.glb at root
        scene_path = os.path.join(self.scene_dir, path)
        if path and os.path.exists(scene_path):
            return scene_path

        # Everything else from the viewer directory
        return os.path.join(self.viewer_dir, path)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

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


def main():
    parser = argparse.ArgumentParser(description='Serve the WebXR aorta viewer')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--scene-dir', default='viz/output',
                        help='Directory containing exported scene (with scene.json and aorta.glb)')
    args = parser.parse_args()

    viewer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viewer')
    scene_dir = os.path.abspath(args.scene_dir)

    handler = functools.partial(
        ViewerHandler,
        viewer_dir=viewer_dir,
        scene_dir=scene_dir,
    )

    ip = get_local_ip()
    with socketserver.TCPServer(('', args.port), handler) as httpd:
        print(f'Viewer:  http://localhost:{args.port}')
        print(f'Mobile:  http://{ip}:{args.port}')
        print(f'Scenes:  {scene_dir}')
        print('Press Ctrl+C to stop')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nStopped.')


if __name__ == '__main__':
    main()
