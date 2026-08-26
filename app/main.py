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

# main.py อยู่ที่:
# /app/app/main.py
#
# static อยู่ที่:
# /app/static/
#
# ดังนั้นต้องถอยกลับ 1 ระดับ

BASE_DIR = Path(__file__).resolve().parent.parent

STATIC_DIR = BASE_DIR / "static"


# =========================================================
# Whisper
# =========================================================

WHISPER_BIN = os.getenv(
    "WHISPER_BIN",
    "/opt/whisper.cpp/build/bin/whisper-cli"
)

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "/opt/whisper.cpp/models/ggml-tiny.bin"
)


# =========================================================
# OpenAI
# =========================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    ""
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)


# =========================================================
# FastAPI
# =========================================================

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
        StaticFiles(
            directory=str(STATIC_DIR)
        ),
        name="static"
    )


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    index_file = STATIC_DIR / "index.html"

    if index_file.exists():

        return FileResponse(
            str(index_file),
            media_type="text/html"
        )

    return {
        "status": "ok",
        "service": "AI Thai Dub V2.0",
        "error": "ไม่พบ static/index.html",
        "base_dir": str(BASE_DIR),
        "static_dir": str(STATIC_DIR)
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():

    ffmpeg_path = shutil.which("ffmpeg")

    return {

        "status": "ok",

        "service": "AI Thai Dub V2.0",

        "version": "2.0",

        "whisper_exists":
            os.path.exists(WHISPER_BIN),

        "whisper_executable":
            os.access(
                WHISPER_BIN,
                os.X_OK
            ) if os.path.exists(WHISPER_BIN) else False,

        "model_exists":
            os.path.exists(WHISPER_MODEL),

        "model_size":
            (
                os.path.getsize(
                    WHISPER_MODEL
                )
                if os.path.exists(WHISPER_MODEL)
                else 0
            ),

        "ffmpeg_exists":
            ffmpeg_path is not None,

        "ffmpeg_path":
            ffmpeg_path,

        "static_exists":
            STATIC_DIR.exists(),

        "index_exists":
            (
                STATIC_DIR / "index.html"
            ).exists(),

        "openai_configured":
            bool(OPENAI_API_KEY)
    }


# =========================================================
# WAV CHECK
# =========================================================

def check_wav(
    wav_file: Path
):

    try:

        with wave.open(
            str(wav_file),
            "rb"
        ) as wav:

            channels = wav.getnchannels()

            sample_width = wav.getsampwidth()

            sample_rate = wav.getframerate()

            frames = wav.getnframes()

        return {

            "channels":
                channels,

            "sample_width":
                sample_width,

            "sample_rate":
                sample_rate,

            "frames":
                frames
        }

    except Exception as e:

        raise RuntimeError(
            f"WAV ตรวจสอบไม่ได้: {str(e)}"
        )


# =========================================================
# WHISPER TRANSCRIPTION
# =========================================================

def run_whisper(
    audio_file: Path
):

    if not os.path.exists(
        WHISPER_BIN
    ):

        raise RuntimeError(
            "ไม่พบ Whisper binary: "
            + WHISPER_BIN
        )


    if not os.path.exists(
        WHISPER_MODEL
    ):

        raise RuntimeError(
            "ไม่พบ Whisper model: "
            + WHISPER_MODEL
        )


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


    print("=" * 60)

    print("WHISPER START")

    print("=" * 60)

    print(
        "COMMAND:",
        " ".join(whisper_cmd)
    )


    result = subprocess.run(

        whisper_cmd,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,

        timeout=300
    )


    print(
        "RETURN CODE:",
        result.returncode
    )


    print(
        "STDOUT:"
    )

    print(
        result.stdout[-10000:]
    )


    print(
        "STDERR:"
    )

    print(
        result.stderr[-10000:]
    )


    if result.returncode != 0:

        raise RuntimeError(

            "Whisper ทำงานไม่สำเร็จ\n\n"

            f"Return code: "
            f"{result.returncode}\n\n"

            "STDOUT:\n"
            f"{result.stdout[-4000:]}\n\n"

            "STDERR:\n"
            f"{result.stderr[-4000:]}"
        )


    raw_text = (
        result.stdout
        .strip()
    )


    if not raw_text:

        return "ไม่พบข้อความเสียงพูด"


    # -----------------------------------------------------
    # Clean Whisper output
    # -----------------------------------------------------

    lines = []


    for line in raw_text.splitlines():

        line = line.strip()


        if not line:
            continue


        if line.startswith(
            "whisper_"
        ):
            continue


        if line.startswith(
            "main:"
        ):
            continue


        if line.startswith(
            "system_info:"
        ):
            continue


        if line.startswith(
            "ggml_"
        ):
            continue


        if line.startswith(
            "whisper_print"
        ):
            continue


        lines.append(line)


    transcript = "\n".join(
        lines
    ).strip()


    if not transcript:

        transcript = raw_text


    return transcript


# =========================================================
# TRANSCRIBE API
# =========================================================

@app.post("/api/transcribe")
@app.post("/api/upload")
async def transcribe(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # File check
    # -----------------------------------------------------

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
        ".m4a",

        ".aac",
        ".flac",
        ".ogg"
    }


    if suffix not in allowed:

        raise HTTPException(

            status_code=400,

            detail=(
                "รองรับ MP4, MOV, M4V, AVI, MKV, "
                "WEBM, MP3, WAV, M4A, AAC, FLAC และ OGG"
            )
        )


    if shutil.which("ffmpeg") is None:

        raise HTTPException(

            status_code=500,

            detail="ไม่พบ FFmpeg"
        )


    if not os.path.exists(
        WHISPER_BIN
    ):

        raise HTTPException(

            status_code=500,

            detail="ไม่พบ Whisper binary"
        )


    if not os.path.exists(
        WHISPER_MODEL
    ):

        raise HTTPException(

            status_code=500,

            detail="ไม่พบ Whisper model"
        )


    # =====================================================
    # Temporary workspace
    # =====================================================

    try:

        with tempfile.TemporaryDirectory() as temp:

            temp_dir = Path(temp)


            input_file = (
                temp_dir
                / f"input{suffix}"
            )


            audio_file = (
                temp_dir
                / "audio.wav"
            )


            # -------------------------------------------------
            # Save uploaded file
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


            input_size = (
                input_file.stat().st_size
            )


            if input_size <= 0:

                raise HTTPException(

                    status_code=400,

                    detail="ไฟล์ว่าง"
                )


            print("=" * 60)

            print("UPLOAD")

            print("=" * 60)

            print(
                "Filename:",
                file.filename
            )

            print(
                "Size:",
                input_size
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


            print("=" * 60)

            print("FFMPEG START")

            print("=" * 60)


            ffmpeg_result = subprocess.run(

                ffmpeg_cmd,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True,

                timeout=180
            )


            print(
                "FFMPEG RETURN CODE:",
                ffmpeg_result.returncode
            )


            if ffmpeg_result.returncode != 0:

                raise HTTPException(

                    status_code=500,

                    detail=(

                        "FFmpeg ไม่สามารถแปลงเสียงได้\n\n"

                        + ffmpeg_result.stderr[-3000:]
                    )
                )


            if not audio_file.exists():

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "FFmpeg ไม่ได้สร้าง audio.wav"
                    )
                )


            # =================================================
            # WAV validation
            # =================================================

            wav_info = check_wav(
                audio_file
            )


            print(
                "WAV INFO:",
                wav_info
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

            transcript = run_whisper(
                audio_file
            )


            duration_seconds = round(

                wav_info["frames"]
                /
                wav_info["sample_rate"],

                2
            )


            print("=" * 60)

            print("TRANSCRIPTION COMPLETE")

            print("=" * 60)

            print(
                transcript
            )


            # =================================================
            # Response
            # =================================================

            return {

                "status":
                    "ok",

                "version":
                    "2.0",

                "filename":
                    file.filename,

                "transcript":
                    transcript,

                "audio": {

                    "sample_rate":
                        wav_info["sample_rate"],

                    "channels":
                        wav_info["channels"],

                    "duration_seconds":
                        duration_seconds
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

        print(
            "UNEXPECTED ERROR:",
            repr(e)
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "เกิดข้อผิดพลาด:\n"
                + repr(e)
            )
        )


# =========================================================
# OPENAI TRANSLATION MODEL
# =========================================================

class TranslateRequest(BaseModel):

    text: str

    target_language: str = "Thai"


# =========================================================
# TRANSLATE WITH OPENAI
# =========================================================

def translate_with_openai(
    text: str,
    target_language: str = "Thai"
):

    if not OPENAI_API_KEY:

        raise RuntimeError(
            "ยังไม่ได้ตั้งค่า OPENAI_API_KEY "
            "ใน Render Environment Variables"
        )


    if not text.strip():

        return ""


    url = (
        "https://api.openai.com/v1/chat/completions"
    )


    payload = {

        "model":
            OPENAI_MODEL,

        "messages": [

            {
                "role": "system",

                "content": (
                    "You are a professional video "
                    "translation assistant. "
                    "Translate the provided transcript "
                    "into natural Thai. "
                    "Preserve the original meaning. "
                    "Do not add explanations. "
                    "Return only the translated text."
                )
            },

            {

                "role": "user",

                "content": (
                    f"Translate this transcript "
                    f"to {target_language}:\n\n"
                    f"{text}"
                )
            }
        ],

        "temperature":
            0.2
    }


    data = json.dumps(
        payload
    ).encode(
        "utf-8"
    )


    request = urllib.request.Request(

        url,

        data=data,

        method="POST",

        headers={

            "Content-Type":
                "application/json",

            "Authorization":
                "Bearer "
                + OPENAI_API_KEY
        }
    )


    try:

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            response_data = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )


    except urllib.error.HTTPError as e:

        error_body = ""

        try:

            error_body = (
                e.read()
                .decode("utf-8")
            )

        except Exception:
            pass


        raise RuntimeError(

            "OpenAI API Error "
            f"{e.code}: "
            f"{error_body[-3000:]}"
        )


    except Exception as e:

        raise RuntimeError(
            f"เชื่อมต่อ OpenAI ไม่สำเร็จ: {str(e)}"
        )


    try:

        translated = (

            response_data
            ["choices"]
            [0]
            ["message"]
            ["content"]
            .strip()
        )


    except Exception:

        raise RuntimeError(

            "รูปแบบข้อมูลจาก OpenAI "
            "ไม่ถูกต้อง"
        )


    return translated


# =========================================================
# TRANSLATE API
# =========================================================

@app.post("/api/translate")
async def translate(
    request: TranslateRequest
):

    text = request.text.strip()


    if not text:

        raise HTTPException(

            status_code=400,

            detail="ไม่มีข้อความสำหรับแปล"
        )


    try:

        translated = translate_with_openai(

            text,

            request.target_language
        )


        return {

            "status":
                "ok",

            "version":
                "2.0",

            "source_text":
                text,

            "target_language":
                request.target_language,

            "translated_text":
                translated
        }


    except Exception as e:

        print(
            "TRANSLATION ERROR:",
            repr(e)
        )


        raise HTTPException(

            status_code=500,

            detail=str(e)
        )


# =========================================================
# ROOT TEST
# =========================================================

@app.get("/api")
def api_info():

    return {

        "status":
            "ok",

        "service":
            "AI Thai Dub V2.0",

        "endpoints": {

            "home":
                "/",

            "health":
                "/api/health",

            "transcribe":
                "/api/transcribe",

            "translate":
                "/api/translate"
        }
    }
