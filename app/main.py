from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os
import subprocess
import tempfile
import shutil
import wave
import json
import urllib.request
import urllib.error


# =========================================================
# AI Thai Dub V2.0
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

WHISPER_BIN = "/opt/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/opt/whisper.cpp/models/ggml-tiny.bin"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-mini")


app = FastAPI(
    title="AI Thai Dub V2.0",
    version="2.0"
)


# =========================================================
# Static Website
# =========================================================

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
        "service": "AI Thai Dub V2.0"
    }


# =========================================================
# Health
# =========================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "AI Thai Dub V2.0",

        "whisper_exists":
            os.path.exists(WHISPER_BIN),

        "model_exists":
            os.path.exists(WHISPER_MODEL),

        "ffmpeg_exists":
            shutil.which("ffmpeg") is not None,

        "openai_configured":
            bool(OPENAI_API_KEY),

        "static_exists":
            STATIC_DIR.exists(),

        "index_exists":
            (STATIC_DIR / "index.html").exists()
    }


# =========================================================
# WAV CHECK
# =========================================================

def check_wav(wav_file):

    try:

        with wave.open(
            str(wav_file),
            "rb"
        ) as wav:

            return {
                "channels":
                    wav.getnchannels(),

                "sample_width":
                    wav.getsampwidth(),

                "sample_rate":
                    wav.getframerate(),

                "frames":
                    wav.getnframes()
            }

    except Exception as e:

        raise RuntimeError(
            f"WAV ตรวจสอบไม่ได้: {str(e)}"
        )


# =========================================================
# TRANSCRIBE
# =========================================================

@app.post("/api/transcribe")
@app.post("/api/upload")
async def transcribe(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="ไม่ได้เลือกไฟล์"
        )


    suffix = Path(
        file.filename
    ).suffix.lower()


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
            detail=(
                "รองรับ MP4, MOV, M4V, AVI, MKV, "
                "WEBM, MP3, WAV และ M4A"
            )
        )


    if not os.path.exists(WHISPER_BIN):

        raise HTTPException(
            status_code=500,
            detail="ไม่พบ Whisper binary"
        )


    if not os.path.exists(WHISPER_MODEL):

        raise HTTPException(
            status_code=500,
            detail="ไม่พบ Whisper model"
        )


    if shutil.which("ffmpeg") is None:

        raise HTTPException(
            status_code=500,
            detail="ไม่พบ FFmpeg"
        )


    try:

        with tempfile.TemporaryDirectory() as temp:

            temp_dir = Path(temp)

            input_file = (
                temp_dir / f"input{suffix}"
            )

            audio_file = (
                temp_dir / "audio.wav"
            )


            # -------------------------------------------------
            # Save upload
            # -------------------------------------------------

            with open(
                input_file,
                "wb"
            ) as output:

                while True:

                    chunk = await file.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    output.write(chunk)


            if input_file.stat().st_size <= 0:

                raise HTTPException(
                    status_code=400,
                    detail="ไฟล์ว่าง"
                )


            # =================================================
            # FFmpeg
            # =================================================

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


            result = subprocess.run(

                ffmpeg_cmd,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True,

                timeout=180
            )


            if result.returncode != 0:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "FFmpeg ไม่สามารถแปลงเสียงได้\n\n"
                        + result.stderr[-3000:]
                    )
                )


            if not audio_file.exists():

                raise HTTPException(

                    status_code=500,

                    detail="ไม่พบ audio.wav"
                )


            # =================================================
            # WAV
            # =================================================

            wav_info = check_wav(
                audio_file
            )


            if wav_info["channels"] != 1:

                raise HTTPException(

                    status_code=500,

                    detail="WAV ต้องเป็น Mono"
                )


            if wav_info["sample_rate"] != 16000:

                raise HTTPException(

                    status_code=500,

                    detail="WAV ต้องเป็น 16000 Hz"
                )


            if wav_info["sample_width"] != 2:

                raise HTTPException(

                    status_code=500,

                    detail="WAV ต้องเป็น 16-bit"
                )


            # =================================================
            # WHISPER
            # =================================================

            whisper_cmd = [

                WHISPER_BIN,

                "-m",
                WHISPER_MODEL,

                "-f",
                str(audio_file),

                "-l",
                "th",

                "-t",
                "2",

                "-nt"
            ]


            whisper = subprocess.run(

                whisper_cmd,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True,

                timeout=300
            )


            if whisper.returncode != 0:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "Whisper ทำงานไม่สำเร็จ\n\n"
                        f"Return code: "
                        f"{whisper.returncode}\n\n"
                        f"{whisper.stderr[-4000:]}"
                    )
                )


            transcript = (
                whisper.stdout.strip()
            )


            # =================================================
            # Clean Whisper Output
            # =================================================

            lines = []

            for line in transcript.splitlines():

                line = line.strip()

                if not line:
                    continue

                lower = line.lower()

                if lower.startswith(
                    (
                        "whisper_",
                        "main:",
                        "system_info:",
                        "ggml_"
                    )
                ):
                    continue

                lines.append(line)


            clean_transcript = "\n".join(
                lines
            ).strip()


            if not clean_transcript:

                clean_transcript = (
                    "ไม่พบข้อความเสียงพูด"
                )


            duration = round(

                wav_info["frames"]
                / wav_info["sample_rate"],

                2
            )


            return {

                "status": "ok",

                "version": "2.0",

                "filename":
                    file.filename,

                "transcript":
                    clean_transcript,

                "audio": {

                    "sample_rate":
                        wav_info["sample_rate"],

                    "channels":
                        wav_info["channels"],

                    "duration_seconds":
                        duration
                }
            }


    except HTTPException:
        raise


    except subprocess.TimeoutExpired:

        raise HTTPException(

            status_code=500,

            detail=(
                "การประมวลผลใช้เวลานานเกินไป "
                "กรุณาลองวิดีโอที่สั้นลง"
            )
        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"เกิดข้อผิดพลาด: {repr(e)}"
        )


# =========================================================
# TRANSLATION REQUEST
# =========================================================

class TranslateRequest(BaseModel):

    text: str

    source_language: str = "auto"

    target_language: str = "th"

    style: str = "natural"


# =========================================================
# AI TRANSLATION
# =========================================================

@app.post("/api/translate")
def translate(
    request: TranslateRequest
):

    text = request.text.strip()


    if not text:

        raise HTTPException(

            status_code=400,

            detail="ไม่มีข้อความสำหรับแปล"
        )


    if len(text) > 30000:

        raise HTTPException(

            status_code=400,

            detail=(
                "ข้อความยาวเกินไป "
                "กรุณาแบ่งข้อความก่อน"
            )
        )


    # ---------------------------------------------------------
    # API KEY
    # ---------------------------------------------------------

    if not OPENAI_API_KEY:

        raise HTTPException(

            status_code=500,

            detail=(
                "ยังไม่ได้ตั้ง OPENAI_API_KEY "
                "ใน Render Environment Variables"
            )
        )


    # =========================================================
    # Prompt
    # =========================================================

    prompt = f"""
คุณคือผู้เชี่ยวชาญด้านการแปลบทพูดสำหรับวิดีโอ

แปลข้อความต่อไปนี้เป็นภาษาไทย

เป้าหมาย:
- ภาษาไทยเป็นธรรมชาติ
- ฟังเหมือนคนไทยพูดจริง
- ไม่แปลตรงตัวจนแข็ง
- รักษาความหมายเดิม
- ไม่เพิ่มข้อมูลที่ไม่มีในต้นฉบับ
- เหมาะสำหรับนำไปทำเสียงพากย์
- รักษาการแบ่งบรรทัดเดิมเท่าที่ทำได้
- ไม่ใส่คำอธิบายเพิ่มเติม
- ตอบเฉพาะข้อความภาษาไทย

สไตล์:
{request.style}

ต้นฉบับ:
{text}
"""


    # =========================================================
    # OpenAI Responses API
    # =========================================================

    payload = {

        "model":
            OPENAI_MODEL,

        "input":
            prompt
    }


    data = json.dumps(
        payload
    ).encode("utf-8")


    req = urllib.request.Request(

        "https://api.openai.com/v1/responses",

        data=data,

        headers={

            "Content-Type":
                "application/json",

            "Authorization":
                f"Bearer {OPENAI_API_KEY}"
        },

        method="POST"
    )


    try:

        with urllib.request.urlopen(
            req,
            timeout=120
        ) as response:

            raw = response.read().decode(
                "utf-8"
            )

            result = json.loads(raw)


    except urllib.error.HTTPError as e:

        error_body = e.read().decode(
            "utf-8",
            errors="ignore"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "OpenAI API Error\n\n"
                + error_body[:4000]
            )
        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "เชื่อมต่อ AI ไม่สำเร็จ\n\n"
                + repr(e)
            )
        )


    # =========================================================
    # Extract Response Text
    # =========================================================

    translated = ""


    if isinstance(result, dict):

        if result.get("output_text"):

            translated = (
                result["output_text"]
            )

        else:

            output = result.get(
                "output",
                []
            )

            for item in output:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                for content in item.get(
                    "content",
                    []
                ):

                    if (
                        isinstance(
                            content,
                            dict
                        )
                        and
                        content.get("type")
                        == "output_text"
                    ):

                        translated += (
                            content.get(
                                "text",
                                ""
                            )
                        )


    translated = translated.strip()


    if not translated:

        raise HTTPException(

            status_code=500,

            detail=(
                "AI ไม่ส่งข้อความแปลกลับมา"
            )
        )


    return {

        "status": "ok",

        "source_language":
            request.source_language,

        "target_language":
            request.target_language,

        "style":
            request.style,

        "translation":
            translated
    }


# =========================================================
# TEXT DOWNLOAD
# =========================================================

@app.post("/api/download-text")
def download_text(data: dict):

    text = str(
        data.get(
            "text",
            ""
        )
    ).strip()


    if not text:

        raise HTTPException(

            status_code=400,

            detail="ไม่มีข้อความ"
        )


    temp = tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".txt",

        mode="w",

        encoding="utf-8"
    )


    try:

        temp.write(text)

        temp.close()


        return FileResponse(

            temp.name,

            media_type="text/plain",

            filename="thai-dub-script.txt"
        )


    except Exception:

        try:
            os.unlink(temp.name)
        except Exception:
            pass

        raise


# =========================================================
# Root API test
# =========================================================

@app.get("/api/version")
def version():

    return {

        "service":
            "AI Thai Dub",

        "version":
            "2.0",

        "features": [

            "video_upload",

            "ffmpeg_audio_extract",

            "whisper_transcription",

            "editable_transcript",

            "ai_thai_translation",

            "text_download",

            "ready_for_tts"
        ]
    }
