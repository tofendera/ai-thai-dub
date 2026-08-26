FROM python:3.11-slim

# =========================================================
# AI Thai Dub V1.8
# Dockerfile
# =========================================================

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# =========================================================
# System packages
# =========================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    ffmpeg \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# =========================================================
# Build whisper.cpp
# =========================================================

RUN git clone --depth 1 \
    https://github.com/ggml-org/whisper.cpp.git \
    /opt/whisper.cpp


# =========================================================
# Build Whisper CLI
#
# IMPORTANT:
# GGML_NATIVE=OFF
# เพื่อไม่ให้ binary ผูกกับ CPU เฉพาะเครื่องตอน build
# =========================================================

RUN cd /opt/whisper.cpp && \
    rm -rf build && \
    SOURCE_DATE_EPOCH=1 cmake \
    -S . \
    -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=OFF \
    -DGGML_AVX2=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON && \
    cmake \
    --build build \
    --config Release \
    --target whisper-cli \
    -j2


# =========================================================
# Check Whisper binary
# =========================================================

RUN test -x /opt/whisper.cpp/build/bin/whisper-cli && \
    /opt/whisper.cpp/build/bin/whisper-cli --help > /dev/null


# =========================================================
# Download multilingual Whisper tiny model
# =========================================================

RUN mkdir -p /opt/whisper.cpp/models && \
    wget -q \
    --show-progress \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin \
    -O /opt/whisper.cpp/models/ggml-tiny.bin


# =========================================================
# Verify model
# =========================================================

RUN test -s /opt/whisper.cpp/models/ggml-tiny.bin


# =========================================================
# Python requirements
# =========================================================

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt


# =========================================================
# Application
# =========================================================

COPY . .


# =========================================================
# Render port
# =========================================================

EXPOSE 10000


# =========================================================
# Start
# =========================================================

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
