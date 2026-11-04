FROM ghcr.io/nvidia/deepstream@sha256:5ee7458d5e795592e0cb182457a7f8281f2c1da7bb38ce4b8506ac693f266746
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STRAW_RUNTIME=native \
    STRAW_BASE_IMAGE=sha256:5ee7458d5e795592e0cb182457a7f8281f2c1da7bb38ce4b8506ac693f266746 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins \
    LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}
COPY requirements/deepstream-demo.txt /opt/straw/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /opt/straw/requirements.txt
WORKDIR /workspace
COPY apps/stream_demo/ /workspace/apps/stream_demo/
RUN g++ -std=c++17 -O2 apps/stream_demo/capture_probe.cpp -o /opt/straw/capture_probe \
    $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0)
COPY configs/straw_realtime_stream512_pose_v1.json /workspace/configs/
COPY docker/demo/model/ /opt/straw/model/
COPY .docker/model/b1_rawrgb512.onnx /opt/straw/model/
COPY .docker/samples/demo_samples/ /workspace/demo_samples/
EXPOSE 8033
VOLUME ["/workspace/artifacts"]
HEALTHCHECK --interval=5s --timeout=4s --start-period=20m --retries=6 \
    CMD python3 -c "import json,urllib.request; s=json.load(urllib.request.urlopen('http://127.0.0.1:8033/health',timeout=3)); exit(0 if s['runtime_verified'] else 1)"
ENTRYPOINT ["python3", "/workspace/apps/stream_demo/container_boot.py"]
