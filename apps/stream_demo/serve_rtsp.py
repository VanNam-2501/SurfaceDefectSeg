"""Local, TCP-only RTSP replay. No labels are rendered into model input."""
import argparse
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import Gst, GstRtspServer, GstRtsp, GLib


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video', type=Path)
    p.add_argument('--port', default='8554')
    args = p.parse_args()
    video = args.video.resolve(strict=True)
    if any(c in str(video) for c in ('"', '\\', '\n')):
        p.error('Unsupported video path')
    Gst.init(None)
    server = GstRtspServer.RTSPServer()
    server.set_address('127.0.0.1')
    server.set_service(args.port)
    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch(f'( filesrc location="{video}" ! qtdemux ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )')
    factory.set_shared(True)
    factory.set_protocols(GstRtsp.RTSPLowerTrans.TCP)
    server.get_mount_points().add_factory('/straw', factory)
    if not server.attach(None):
        raise RuntimeError('Cannot bind RTSP server')
    print(f'RTSP READY rtsp://127.0.0.1:{args.port}/straw', flush=True)
    try:
        GLib.MainLoop().run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
