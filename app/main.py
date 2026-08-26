from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import os
import subprocess
import tempfile


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Thai Dub V1.7")


# =========================
# Static
# =========================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static"
    )


# =========================
# OpenAI
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Home
# =========================

@app.get("/")
async def home():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return {
            "status": "ok",
            "service": "AI Thai Dub V1.7"
        }

    return FileResponse(index_file)


# =========================
# Health
# =========================

@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "service": "AI Thai Dub V1.7",
        "openai_configured": bool(OPENAI_API_KEY)
    }


# =========================
# Transcription
# =========================

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="กรุณาเลือกไฟล์"
        )

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า OPENAI_API_KEY"
        )

    allowed_extensions = {
        ".mp3",
        ".wav",
        ".m4a",
        ".mp4",
        ".mov",
        ".webm",
        ".ogg",
        ".mpeg",
        ".mpga"
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="ไฟล์ชนิดนี้ยังไม่รองรับ"
        )

    contents = await file.read()

    # V1.7 จำกัด 25 MB
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="ไฟล์ใหญ่เกิน 25 MB ใน V1.7"
        )

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            input_file = temp_dir / file.filename
            audio_file = temp_dir / "audio.mp3"

            # บันทึกไฟล์ต้นฉบับ
            with open(input_file, "wb") as f:
                f.write(contents)

            # =========================
            # FFmpeg
            # =========================

            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_file),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "4",
                str(audio_file)
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if result.returncode != 0:

                print(
                    "FFMPEG ERROR:",
                    result.stderr.decode(
                        errors="ignore"
                    )
                )

                raise HTTPException(
                    status_code=500,
                    detail="FFmpeg ไม่สามารถแยกเสียงได้"
                )

            # =========================
            # OpenAI
            # =========================

            with open(audio_file, "rb") as audio:

                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio
                )

            text = transcription.text

            return {
                "status": "success",
                "service": "AI Thai Dub V1.7",
                "filename": file.filename,
                "transcript": text
            }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "TRANSCRIPTION ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาดในการถอดเสียง: {str(e)}"
        )
