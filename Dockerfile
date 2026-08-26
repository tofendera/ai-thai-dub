FROM python:3.11-slim

WORKDIR /app

# ==========================================
# System dependencies
# ==========================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    ffmpeg \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# ==========================================
# Build whisper.cpp
# CPU compatible build
# ==========================================

RUN git clone --depth 1 \
    https://github.com/ggml-org/whisper.cpp.git \
    /opt/whisper.cpp

RUN mkdir -p /opt/whisper.cpp/build && \
    cd /opt/whisper.cpp && \
    SOURCE_DATE_EPOCH=1234567890 \
    cmake -S . \
    -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=OFF \
    -DGGML_AVX2=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DGGML_BMI2=OFF \
    -DGGML_AVX512=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON \
    && \
    cmake --build build --config Release -j1


# ==========================================
# Download Whisper model
# ==========================================

RUN mkdir -p /opt/whisper.cpp/models && \
    wget -q \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin \
    -O /opt/whisper.cpp/models/ggml-tiny-q5_1.bin


# ==========================================
# Python dependencies
# ==========================================

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# ==========================================
# Application
# ==========================================

COPY . .


# ==========================================
# Render
# ==========================================

EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
