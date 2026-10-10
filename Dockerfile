# syntax=docker/dockerfile:1
# ComfyUI Docker Image with Blackwell NVFP4 Support
# Optimized for NVIDIA Blackwell GPUs (RTX 50 series)

# Build arguments for version pinning --Global Scope--
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04
ARG TORCH_WHEEL_URL=https://download.pytorch.org/whl/cu130/torch-2.10.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl
ARG TORCHVISION_WHEEL_URL=https://download.pytorch.org/whl/cu130/torchvision-0.25.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl
ARG TORCHAUDIO_WHEEL_URL=https://download.pytorch.org/whl/cu130/torchaudio-2.10.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl
ARG COMFYUI_BRANCH=master
ARG SAGEATTENTION_VERSION=v2
ARG SAGEATTENTION_USE=1
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG RUN_UID=1000
ARG RUN_GID=1000
# Space separated list of git URLs of custom nodes to bake into the image
# Each URL may pin a tag/branch with a '#ref' suffix
ARG CUSTOM_NODES_URLS=
# Run each cloned node's install.py at build time (1 = yes, 0 = no)
ARG RUN_NODE_INSTALLERS=1

FROM ${CUDA_BASE_IMAGE}
# Re-declaring these ARGs so they are available inside this build stage
ARG RUN_UID
ARG RUN_GID
ARG CUSTOM_NODES_URLS
ARG RUN_NODE_INSTALLERS
ARG SAGEATTENTION_VERSION
ARG SAGEATTENTION_USE
ARG TORCH_WHEEL_URL
ARG TORCHVISION_WHEEL_URL
ARG TORCHAUDIO_WHEEL_URL
ARG COMFYUI_BRANCH
ARG TORCH_CUDA_ARCH_LIST
# Pass these to the runtime
ENV SAGEATTENTION_VERSION=${SAGEATTENTION_VERSION}
ENV SAGEATTENTION_USE=${SAGEATTENTION_USE}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
# Setup environment for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
# Allow pip to install in system Python (we're in a container, it's fine)
# NOTE: do NOT add PIP_IGNORE_INSTALLED=1 here. It makes "pip list" report
# nothing, which breaks ComfyUI-Manager's package detection ("PyTorch is not
# installed" error) and silently disables its torch rollback protection.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
# libgl1, libglib2.0-0: Required for OpenCV (used by many custom nodes)
# ffmpeg: Video processing support
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version.
# --ignore-installed is needed ONLY here: the base image ships pip via apt
# (no RECORD file), so pip cannot uninstall it - installing over it into
# /usr/local works instead. Do NOT set this globally (see note above).
RUN python3 -m pip install --upgrade --ignore-installed pip

# Set working directory
WORKDIR /app

# Clone ComfyUI repository
ARG COMFYUI_BRANCH
RUN git clone --branch ${COMFYUI_BRANCH} https://github.com/comfyanonymous/ComfyUI.git .

# Create the constraints file BEFORE installing PyTorch so that even the
# PyTorch install itself (and everything after it) is constrained.
# This prevents anything from accidentally downgrading/upgrading PyTorch.
RUN echo "torch @ ${TORCH_WHEEL_URL}" > /app/constraints.txt && \
    echo "torchvision @ ${TORCHVISION_WHEEL_URL}" >> /app/constraints.txt && \
    echo "torchaudio @ ${TORCHAUDIO_WHEEL_URL}" >> /app/constraints.txt

# Install PyTorch with CUDA 13.0 support
# These specific wheels are critical for Blackwell GPU support
# Use cache mount for the heavy PyTorch download
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install -c /app/constraints.txt \
    "${TORCH_WHEEL_URL}" \
    "${TORCHVISION_WHEEL_URL}" \
    "${TORCHAUDIO_WHEEL_URL}"

# Extend the constraints with the triton and nvidia-* CUDA package versions
# that the installed torch requires (e.g. torch 2.10.0+cu130 -> triton==3.6.0),
# read live from the installed torch metadata, so no later pip install can
# drift them off the CUDA 13.0 stack.
RUN python3 - <<'EOF' >> /app/constraints.txt
import importlib.metadata as m

seen = set()
for r in m.requires("torch") or []:
    spec = r.split(";", 1)[0].strip()
    name = spec.split("==")[0].strip().lower()
    if (name == "triton" or name.startswith("nvidia-")) and "==" in spec:
        if spec not in seen:
            seen.add(spec)
            print(r.strip())
EOF

# Make the constraints active for EVERY pip run from here on - including
# bare "pip install" calls inside custom node install.py scripts, which
# cannot be given an explicit -c flag. uv does not read PIP_CONSTRAINT,
# so point UV_CONSTRAINT at the same file.
ENV PIP_CONSTRAINT=/app/constraints.txt
ENV UV_CONSTRAINT=/app/constraints.txt

# Install SageAttention for improved attention mechanism performance
# Triton is required for SageAttention's CUDA kernels.
# The version is forced by constraints.txt (must match torch's pin, e.g.
# 3.6.0 for torch 2.10.0+cu130) - installing a newer triton breaks torch.
    RUN if [ "$SAGEATTENTION_VERSION" != "none" ]; then \
      pip install triton -c /app/constraints.txt && \
      pip install packaging ninja setuptools wheel -c /app/constraints.txt && \
      git clone https://github.com/thu-ml/SageAttention.git /app/tmp/sageattention && \
      if [ "$SAGEATTENTION_VERSION" = "v3" ]; then \
        cd /app/tmp/sageattention/sageattention3_blackwell; \
      elif [ "$SAGEATTENTION_VERSION" = "v2" ]; then \
        cd /app/tmp/sageattention; \
      else \
        echo "ERROR: SAGEATTENTION_VERSION must be v2, v3 or none" && exit 1; \
      fi && \
      pip install --no-build-isolation -c /app/constraints.txt . && \
      cd /app && \
      rm -rf /app/tmp/sageattention; \
    fi

# Clone custom nodes listed in CUSTOM_NODES_URLS (space separated git URLs).
# A '#tag' or '#branch' suffix on a URL pins a specific version.
# Cloning happens BEFORE installing any requirements so they can all be
# resolved together in one pass below.
RUN for url in $CUSTOM_NODES_URLS; do \
        ref="${url%%#*}"; \
        frag="${url#*#}"; \
        name="$(basename "${ref}" .git)"; \
        if [ -d "/app/custom_nodes/$name" ]; then \
            echo "Skipping $url - custom_nodes/$name already exists"; \
            continue; \
        fi; \
        echo "Cloning $url into custom_nodes/$name"; \
        if [ "$frag" != "$url" ]; then \
            git clone --depth 1 --branch "$frag" "$ref" "/app/custom_nodes/$name" || echo "WARNING: failed to clone $url"; \
        else \
            git clone --depth 1 "$ref" "/app/custom_nodes/$name" || echo "WARNING: failed to clone $url"; \
        fi; \
    done

# Install additional wheels from wheels.txt if provided
# This is where Nunchaku and other special packages go
# The wheels.txt file should be in the build context
RUN --mount=type=bind,source=.,target=/mnt/context,ro \
    if [ -f "/mnt/context/wheels.txt" ]; then \
        echo "Found wheels.txt, installing wheels..."; \
        while IFS= read -r wheel_url || [ -n "$wheel_url" ]; do \
            # Skip comments and empty lines
            case "$wheel_url" in \
                \#*|"") continue ;; \
            esac; \
            echo "Installing: $wheel_url"; \
            pip install "$wheel_url" -c /app/constraints.txt || true; \
        done < /mnt/context/wheels.txt; \
    else \
        echo "No wheels.txt found, skipping additional wheels..."; \
    fi

# Install ComfyUI's core requirements TOGETHER with every custom node's
# requirements.txt (both URL-cloned nodes and nodes already present in the
# build context) in a SINGLE pip resolution pass.
# Resolving everything at once lets pip reconcile version conflicts between
# ComfyUI and the nodes, instead of the last requirements.txt silently
# overwriting what an earlier install chose. constraints.txt keeps
# torch/torchvision/torchaudio pinned no matter what the nodes ask for.
# If the joint resolution fails, fall back to: core first, then each node
# individually, so one broken node cannot take the whole build down.
RUN --mount=type=bind,source=.,target=/mnt/context,ro \
    set -- -r /app/requirements.txt; \
    for f in /app/custom_nodes/*/requirements.txt /mnt/context/custom_nodes/*/requirements.txt; do \
        if [ -f "$f" ]; then set -- "$@" -r "$f"; fi; \
    done; \
    if pip install "$@" -c /app/constraints.txt; then \
        echo "ComfyUI + custom node requirements resolved together successfully"; \
    else \
        echo "Joint resolution failed - installing core requirements then each node separately"; \
        pip install -r /app/requirements.txt -c /app/constraints.txt; \
        for f in /app/custom_nodes/*/requirements.txt /mnt/context/custom_nodes/*/requirements.txt; do \
            if [ -f "$f" ]; then \
                echo "Installing deps from $f"; \
                pip install -r "$f" -c /app/constraints.txt || echo "WARNING: failed to install $f (continuing)"; \
            fi; \
        done; \
    fi

# Freeze the versions of ABI-sensitive packages that torch's metadata does
# NOT declare. torch 2.10 wheels are compiled against the numpy 2.x C ABI:
# a single node requiring e.g. "numpy<2" would be a valid pip resolution but
# break torch at runtime ("_ARRAY_API not found"). Same story for Pillow,
# opencv and scipy. Freezing them AFTER the joint resolution above means the
# version ComfyUI itself chose becomes the pin for everything that runs
# later: fallback installs, build-time/startup-time install.py scripts and
# ComfyUI-Manager installs at runtime.
RUN python3 - <<'EOF' >> /app/constraints.txt
import importlib.metadata as m

for pkg in ("numpy", "Pillow", "opencv-python-headless", "scipy"):
    try:
        print(f"{pkg}=={m.version(pkg)}")
    except m.PackageNotFoundError:
        pass
EOF

# Run install.py scripts for the cloned custom nodes at build time.
# This bakes in whatever they install (apt packages, compiled wheels,
# downloads). A timeout stops a hanging installer from wedging the build,
# and a failure never fails the build - startup.sh retries every installer
# at container start anyway. Disable with RUN_NODE_INSTALLERS=0.
RUN if [ "$RUN_NODE_INSTALLERS" = "1" ]; then \
        for dir in /app/custom_nodes/*/; do \
            if [ -f "${dir}install.py" ]; then \
                echo "Running build-time install.py for $(basename "$dir")..."; \
                ( cd "$dir" && timeout 900 python3 install.py ) || echo "WARNING: install.py failed for $(basename "$dir") - will retry at startup"; \
            fi; \
        done; \
    fi

# Stage anything install.py downloaded into /app/models (e.g. Impact-Pack's
# ONNX models) under /opt, so startup.sh can seed the bind-mounted models
# directory with it instead of downloading the same files again on first
# startup.
RUN if [ -n "$(ls -A /app/models 2>/dev/null)" ]; then \
        mkdir -p /opt/models_baked && \
        cp -a /app/models/. /opt/models_baked/ && \
        chown -R ${RUN_UID}:${RUN_GID} /opt/models_baked; \
    fi

# Stage the build-time custom nodes under /opt so startup.sh can copy them
# into the bind-mounted custom_nodes directory (the mount hides the image copy)
RUN if [ -n "$CUSTOM_NODES_URLS" ]; then \
        mkdir -p /opt/custom_nodes_baked; \
        for url in $CUSTOM_NODES_URLS; do \
            ref="${url%%#*}"; \
            name="$(basename "${ref}" .git)"; \
            if [ -d "/app/custom_nodes/$name" ]; then \
                cp -a "/app/custom_nodes/$name" /opt/custom_nodes_baked/; \
            fi; \
        done; \
        chown -R ${RUN_UID}:${RUN_GID} /opt/custom_nodes_baked; \
    fi

# Create startup script that syncs baked custom nodes and runs their installers
# Some custom nodes need post-install setup that happens at runtime
COPY <<'EOF' /app/startup.sh
#!/bin/bash
set -e

# Protect PyTorch versions from being changed
export PIP_CONSTRAINT=/app/constraints.txt

# Copy custom nodes baked into the image at build time (CUSTOM_NODES_URLS)
# into the bind-mounted custom_nodes dir, without overwriting existing files
if [ -d /opt/custom_nodes_baked ]; then
    mkdir -p /app/custom_nodes
    cp -rn /opt/custom_nodes_baked/. /app/custom_nodes/ 2>/dev/null || true
fi

# Seed models that install.py scripts downloaded at build time into the
# bind-mounted models dir (no overwrite), so the installers below find the
# files already present and skip re-downloading them
if [ -d /opt/models_baked ]; then
    mkdir -p /app/models
    cp -rn /opt/models_baked/. /app/models/ 2>/dev/null || true
fi

# Run install.py scripts for any custom nodes that have them
if [ -d "/app/custom_nodes" ]; then
    for dir in /app/custom_nodes/*/; do
        if [ -f "$dir/install.py" ]; then
            echo "Running install script for $(basename $dir)..."
            cd "$dir" && timeout 900 python3 install.py || true
            cd /app
        fi
    done
fi

echo "Starting ComfyUI..."
exec "$@"
EOF
RUN chmod +x /app/startup.sh

# Rootless runtime setup
# Create an unprivileged user matching the host user (RUN_UID/RUN_GID) that owns
# the bind-mounted folders, so ComfyUI never runs as root and created files
# (outputs, workflows, custom nodes) are owned by your host user.
# NOTE: The Ubuntu base image ships with a default 'ubuntu' user at 1000:1000,
# remove it first to avoid UID/GID conflicts.
RUN (userdel -r ubuntu >/dev/null 2>&1 || true) && \
    groupadd -f -g ${RUN_GID} comfyui && \
    useradd -u ${RUN_UID} -g ${RUN_GID} -d /home/comfyui -m -s /bin/bash comfyui && \
    chown -R ${RUN_UID}:${RUN_GID} /app /home/comfyui

# Writable HOME for the runtime user (triton cache, git config, etc.)
ENV HOME=/home/comfyui
ENV XDG_CACHE_HOME=/home/comfyui/.cache
# Give the runtime user ownership of the Python system package dirs so
# runtime pip installs (custom node install.py, ComfyUI-Manager) write
# there directly.
# NOTE: do NOT use PIP_USER=1 for this. User mode makes "pip list" show only
# ~/.local packages (i.e. nothing here), which makes ComfyUI-Manager report
# "PyTorch is not installed" and silently disables its torch rollback
# protection, and it splits packages across two locations at runtime.
RUN chown -R ${RUN_UID}:${RUN_GID} \
      /usr/local/lib/python3.12/dist-packages \
      /usr/local/bin

# Run as the non-root user
USER comfyui

# Expose ComfyUI web interface port
EXPOSE 8188

# Use startup script as entrypoint
ENTRYPOINT ["/app/startup.sh"]

# Default command - can be overridden in docker-compose.yml
CMD if [ "$SAGEATTENTION_USE" != "0" ] && { [ "$SAGEATTENTION_VERSION" = "v2" ] || [ "$SAGEATTENTION_VERSION" = "v3" ]; }; then \
      exec python3 main.py --listen 0.0.0.0 --use-sage-attention; \
    else \
      exec python3 main.py --listen 0.0.0.0; \
    fi
