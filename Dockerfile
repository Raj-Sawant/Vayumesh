# Multi-stage Dockerfile for CPU/GPU builds
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git wget cmake build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3.12 -m venv /venv && /venv/bin/pip install --upgrade pip && /venv/bin/pip install -r requirements.txt

# Copy source
COPY . .

# Runtime stage
FROM ${BASE_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
COPY --from=builder /app /app

WORKDIR /app
ENV PATH="/venv/bin:${PATH}"
ENTRYPOINT ["python3.12", "src/pipeline.py"]