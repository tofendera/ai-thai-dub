from pathlib import Path
import os
import subprocess
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


app = FastAPI(title="AI Thai Dub V1.6")


# -------------------------
# OpenAI
# -------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY
) if OPENAI_API_KEY else None


# -------------------------
# Static Web App
# -------------------------

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static"
    )


@app.get("/")
async def home():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return {
            "status": "ok",
            "message": "AI Thai Dub V1.6 is running"
        }

    return FileResponse(index_file)


# -------------------------
# Health
# -------------------------

@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "service": "AI Thai Dub V1.6",
        "openai_configured": bool(OPENAI_API_KEY)
    }


# -------------------------
# Upload
# -------------------------

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="กรุณาเลือกไฟล์"
        )

    allowed_extensions = {
        ".mp3",
        ".wav",
        ".m4a",
        ".mp4",
        ".mov",
        ".webm",
        ".ogg"
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="รองรับไฟล์ MP3, WAV, M4A, MP4, MOV, WEBM และ OGG"
        )

    file_id = uuid.uuid4().hex

    input_file = UPLOAD_DIR / f"{file_id}{extension}"

    contents = await file.read()

    with open(input_file, "wb") as f:
        f.write(contents)

    return {
        "status": "success",
        "job_id": file_id,
        "filename": file.filename,
        "size": len(contents),
        "message": "อัปโหลดไฟล์สำเร็จ"
    }


# -------------------------
# AI Dubbing
# -------------------------

@app.post("/api/dub/{job_id}")
async def dub_video(job_id: str):

    if not client:
        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า OPENAI_API_KEY"
        )

    # หาไฟล์ต้นฉบับ
    input_files = list(
        UPLOAD_DIR.glob(f"{job_id}.*")
    )

    if not input_files:
        raise HTTPException(
            status_code=404,
            detail="ไม่พบไฟล์"
        )

    input_file = input_files[0]

    try:

        # -------------------------
        # 1. Extract audio
        # -------------------------

        audio_file = UPLOAD_DIR / f"{job_id}.mp3"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_file),
                "-vn",
                "-acodec",
                "libmp3lame",
                "-b:a",
                "128k",
                str(audio_file)
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


        # -------------------------
        # 2. Transcription
        # -------------------------

        with open(audio_file, "rb") as audio:

            transcription = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio
            )

        original_text = transcription.text


        # -------------------------
        # 3. Translate to Thai
        # -------------------------

        translation = client.responses.create(

            model="gpt-5.6-luna",

            input=[
                {
                    "role": "system",
                    "content": (
                        "คุณเป็นนักแปลและเขียนบทพากย์ภาษาไทย "
                        "แปลข้อความให้เป็นภาษาไทยธรรมชาติ "
                        "เหมาะสำหรับการพากย์วิดีโอ "
                        "รักษาความหมายเดิม "
                        "ไม่ต้องอธิบายเพิ่มเติม "
                        "ส่งเฉพาะบทภาษาไทย"
                    )
                },
                {
                    "role": "user",
                    "content": original_text
                }
            ]
        )

        thai_text = translation.output_text.strip()


        # -------------------------
        # 4. Generate Thai voice
        # -------------------------

        # TTS จำกัด input ต่อ request
        # จึงตัดข้อความเป็นช่วงสั้น ๆ

        chunks = []

        max_chars = 3500

        while len(thai_text) > max_chars:

            cut = thai_text.rfind(
                " ",
                0,
                max_chars
            )

            if cut <= 0:
                cut = max_chars

            chunks.append(
                thai_text[:cut]
            )

            thai_text = thai_text[cut:].strip()

        if thai_text:
            chunks.append(thai_text)


        voice_files = []

        for index, chunk in enumerate(chunks):

            voice_file = (
                OUTPUT_DIR /
                f"{job_id}_voice_{index}.mp3"
            )

            speech = client.audio.speech.create(

                model="gpt-4o-mini-tts",

                voice="coral",

                input=chunk,

                instructions=(
                    "พูดภาษาไทยอย่างเป็นธรรมชาติ "
                    "น้ำเสียงชัดเจน เป็นมิตร "
                    "เหมาะสำหรับการพากย์วิดีโอ"
                ),

                response_format="mp3"
            )

            speech.write_to_file(
                voice_file
            )

            voice_files.append(voice_file)


        # -------------------------
        # 5. Merge voice chunks
        # -------------------------

        concat_file = OUTPUT_DIR / f"{job_id}_concat.txt"

        with open(concat_file, "w", encoding="utf-8") as f:

            for voice_file in voice_files:

                f.write(
                    f"file '{voice_file}'\n"
                )


        thai_audio = OUTPUT_DIR / f"{job_id}_thai.mp3"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(thai_audio)
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


        # -------------------------
        # 6. Replace video audio
        # -------------------------

        final_video = (
            OUTPUT_DIR /
            f"{job_id}_thai_dub.mp4"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_file),
                "-i",
                str(thai_audio),

                "-map",
                "0:v:0",
                "-map",
                "1:a:0",

                "-c:v",
                "copy",

                "-c:a",
                "aac",

                "-shortest",

                str(final_video)
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


        # -------------------------
        # 7. Return result
        # -------------------------

        return {
            "status": "success",
            "job_id": job_id,
            "original_text": original_text,
            "thai_text": thai_text,
            "download_url": (
                f"/api/download/{job_id}"
            ),
            "message": "พากย์ภาษาไทยสำเร็จ"
        }


    except subprocess.CalledProcessError as e:

        raise HTTPException(
            status_code=500,
            detail="FFmpeg ประมวลผลไม่สำเร็จ"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -------------------------
# Download
# -------------------------

@app.get("/api/download/{job_id}")
async def download_video(job_id: str):

    output_file = (
        OUTPUT_DIR /
        f"{job_id}_thai_dub.mp4"
    )

    if not output_file.exists():

        raise HTTPException(
            status_code=404,
            detail="ยังไม่มีไฟล์ผลลัพธ์"
        )

    return FileResponse(
        output_file,
        media_type="video/mp4",
        filename="AI-Thai-Dub.mp4"
    )
