FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    ffmpeg \
    wget \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Build whisper.cpp
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git /opt/whisper.cpp && \
    cmake -S /opt/whisper.cpp -B /opt/whisper.cpp/build \
    -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/whisper.cpp/build -j1 --target whisper-cli

# Download small Whisper model
RUN mkdir -p /opt/whisper.cpp/models && \
    wget -q \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin \
    -O /opt/whisper.cpp/models/ggml-tiny-q5_1.bin

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
