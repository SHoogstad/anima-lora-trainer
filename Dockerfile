FROM ubuntu:24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git wget curl ca-certificates software-properties-common \
    python3 python3-venv python3-dev \
    libgl1 libglib2.0-0 tini build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Install Intel GPU runtime stack (no oneAPI) ---
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update && apt-get install -y --no-install-recommends \
        intel-media-va-driver-non-free \
        libmfx-gen1 \
        libvpl2 libvpl-tools \
        libva-glx2 va-driver-all vainfo \
        libze-intel-gpu1 libze1 libze-dev \
        intel-metrics-discovery \
        intel-opencl-icd clinfo \
        intel-gsc intel-ocloc \
    && rm -rf /var/lib/apt/lists/*

# NOTE: The image is intentionally "OS only" — no Python venv, no PyTorch, no
# model weights are baked in. All of that lives on the /workspace volume (see
# docker-compose.yml) and is set up by start.sh on first boot, so rebuilding the
# image never wipes your environment or downloads.
#
# start.sh is kept in /usr/local/bin (the OS layer), NOT in /workspace, so the
# /workspace volume mount cannot shadow it.
COPY docker/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Persistent workspace (mounted as a volume at runtime)
WORKDIR /workspace

# Gradio web UI
EXPOSE 7860

# Set tini as entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["/usr/local/bin/start.sh"]
