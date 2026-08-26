from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import subprocess
import tempfile


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

WHISPER_BIN = "/opt/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/opt/whisper.cpp/models/ggml-tiny-q5_1.bin"

app = FastAPI(title="AI Thai Dub V1.7")


# =========================
# Static website
# =========================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static"
    )


@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {
        "status": "ok",
        "service": "AI Thai Dub V1.7"
    }


# =========================
# Health check
# =========================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "AI Thai Dub V1.7",
        "whisper_exists": os.path.exists(WHISPER_BIN),
        "model_exists": os.path.exists(WHISPER_MODEL)
    }


# =========================
# Whisper transcription
# =========================

@app.post("/api/transcribe")
@app.post("/api/upload")
async def transcribe(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="ไม่ได้เลือกไฟล์"
        )

    suffix = Path(file.filename).suffix.lower()

    allowed = {
        ".mp4",
        ".mov",
        ".m4v",
        ".avi",
        ".mkv",
        ".webm",
        ".mp3",
        ".wav",
        ".m4a"
    }

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail="รองรับไฟล์ MP4, MOV, M4V, AVI, MKV, WEBM, MP3, WAV และ M4A"
        )

    if not os.path.exists(WHISPER_BIN):
        raise HTTPException(
            status_code=500,
            detail="ไม่พบ Whisper"
        )

    if not os.path.exists(WHISPER_MODEL):
        raise HTTPException(
            status_code=500,
            detail="ไม่พบ Whisper model"
        )

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            input_file = temp_dir / f"input{suffix}"
            audio_file = temp_dir / "audio.wav"

            # Save uploaded file
            with open(input_file, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)

                    if not chunk:
                        break

                    f.write(chunk)

            # Convert video/audio to 16 kHz mono WAV
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_file),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(audio_file)
            ]

            ffmpeg_result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )

            if ffmpeg_result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail="ไม่สามารถแปลงไฟล์เสียงได้"
                )

            # Run Whisper
            whisper_cmd = [
                WHISPER_BIN,
                "-m",
                WHISPER_MODEL,
                "-f",
                str(audio_file),
                "-l",
                "auto",
                "-nt"
            ]

            whisper_result = subprocess.run(
                whisper_cmd,
                capture_output=True,
                text=True
            )

            if whisper_result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Whisper transcription failed: "
                        + whisper_result.stderr[-1000:]
                    )
                )

            transcript = whisper_result.stdout.strip()

            if not transcript:
                transcript = "ไม่สามารถตรวจพบเสียงพูดในไฟล์ได้"

            return {
                "status": "ok",
                "filename": file.filename,
                "transcript": transcript
            }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาด: {str(e)}"
        )
