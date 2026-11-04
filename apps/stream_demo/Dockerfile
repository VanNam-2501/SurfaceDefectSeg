FROM ghcr.io/nvidia/deepstream:9.2-samples-multiarch
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*
COPY deepstream_probe.cpp /opt/straw/deepstream_probe.cpp
RUN set -eux; \
    SDK=/opt/nvidia/deepstream/deepstream; \
    test -f "$SDK/sources/includes/gstnvdsinfer.h"; \
    g++ -std=c++17 -O2 /opt/straw/deepstream_probe.cpp -o /opt/straw/deepstream_probe \
    -I"$SDK/sources/includes" -I/usr/local/cuda/include -L"$SDK/lib" -Wl,-rpath,"$SDK/lib" \
    $(pkg-config --cflags --libs gstreamer-1.0) -lnvdsgst_meta -lnvds_meta -pthread
ENTRYPOINT ["/opt/straw/deepstream_probe"]
