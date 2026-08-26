FROM python:3.11-slim

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
        && \
    rm -rf /var/lib/apt/lists/*


# =========================================================
# Build whisper.cpp
# =========================================================

RUN git clone --depth 1 \
    https://github.com/ggml-org/whisper.cpp.git \
    /opt/whisper.cpp


RUN cmake \
    -S /opt/whisper.cpp \
    -B /opt/whisper.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON


RUN cmake \
    --build /opt/whisper.cpp/build \
    --config Release \
    -j1


# =========================================================
# Download multilingual tiny model
# =========================================================

RUN mkdir -p \
    /opt/whisper.cpp/models


RUN wget -q \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin \
    -O /opt/whisper.cpp/models/ggml-tiny.bin


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

CMD [
    "uvicorn",
    "main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "10000"
]
