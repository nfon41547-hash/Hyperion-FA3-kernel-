# =============================================================================
# Hyperion HALO – Production Container
# Base image: NVIDIA PyTorch NGC (Ampere-optimised, CUDA 12.1)
# =============================================================================
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Build arguments
ARG HYPERION_VERSION=3.0.0
ARG PYTHON_VERSION=3.11

LABEL maintainer="Hyperion HALO Team" \
      version="${HYPERION_VERSION}" \
      description="Hyperion HALO – Production LLM Inference Server"

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        libssl-dev \
        ninja-build \
        pciutils \
        numactl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy source
# ---------------------------------------------------------------------------
COPY hyperion_halo_production/ /workspace/hyperion_halo_production/
COPY hyperion_fa3_production_final.py /workspace/
COPY main.py /workspace/
COPY config/ /workspace/config/

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" \
    CUDA_VISIBLE_DEVICES=0 \
    HYPERION_LOG_LEVEL=INFO

# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import hyperion_halo_production; print('ok')" || exit 1

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
COPY hyperion_halo_production/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
