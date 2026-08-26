from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os
import subprocess
import tempfile
import shutil
import wave


# =========================================================
# AI Thai Dub V1.8
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# main.py อยู่ใน /app
# static อยู่ที่ root/static
STATIC_DIR = BASE_DIR.parent / "static"


# =========================================================
# Whisper
# =========================================================

WHISPER_BIN = "/opt/whisper.cpp/build/bin/whisper-cli"

WHISPER_MODEL = "/opt/whisper.cpp/models/ggml-tiny.bin"


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="AI Thai Dub V1.8",
    version="1.8"
)


# =========================================================
# Debug information
# =========================================================

print("=" * 60)
print("AI THAI DUB V1.8")
print("=" * 60)

print("BASE_DIR:")
print(BASE_DIR)

print("STATIC_DIR:")
print(STATIC_DIR)

print("STATIC EXISTS:")
print(STATIC_DIR.exists())

print("INDEX EXISTS:")
print((STATIC_DIR / "index.html").exists())

print("WHISPER BIN:")
print(WHISPER_BIN)

print("WHISPER BIN EXISTS:")
print(os.path.exists(WHISPER_BIN))

print("WHISPER MODEL:")
print(WHISPER_MODEL)

print("WHISPER MODEL EXISTS:")
print(os.path.exists(WHISPER_MODEL))

print("FFMPEG EXISTS:")
print(shutil.which("ffmpeg") is not None)

print("=" * 60)


# =========================================================
# Static website
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
# Home
# =========================================================

@app.get("/")
def home():

    index_file = STATIC_DIR / "index.html"

    print("=" * 60)
    print("HOME REQUEST")
    print("=" * 60)

    print("INDEX FILE:")
    print(index_file)

    print("INDEX EXISTS:")
    print(index_file.exists())

    if index_file.exists():

        return FileResponse(
            str(index_file)
        )

    return {
        "status": "ok",
        "service": "AI Thai Dub V1.8",
        "message": "Static website not found",
        "static_dir": str(STATIC_DIR),
        "index_exists": False
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/api/health")
def health():

    index_file = STATIC_DIR / "index.html"

    return {

        "status": "ok",

        "service": "AI Thai Dub V1.8",

        "version": "1.8",

        "paths": {

            "base_dir":
                str(BASE_DIR),

            "static_dir":
                str(STATIC_DIR),

            "index_file":
                str(index_file)

        },

        "files": {

            "static_exists":
                STATIC_DIR.exists(),

            "index_exists":
                index_file.exists(),

            "whisper_exists":
                os.path.exists(
                    WHISPER_BIN
                ),

            "model_exists":
                os.path.exists(
                    WHISPER_MODEL
                ),

            "ffmpeg_exists":
                shutil.which(
                    "ffmpeg"
                ) is not None

        }

    }


# =========================================================
# Check WAV
# =========================================================

def check_wav(wav_file):

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
# Transcribe
# =========================================================

@app.post("/api/transcribe")
@app.post("/api/upload")
async def transcribe(
    file: UploadFile = File(...)
):

    # =====================================================
    # File check
    # =====================================================

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
                "รองรับ MP4, MOV, M4V, AVI, "
                "MKV, WEBM, MP3, WAV และ M4A"
            )

        )


    # =====================================================
    # Whisper check
    # =====================================================

    if not os.path.exists(
        WHISPER_BIN
    ):

        raise HTTPException(

            status_code=500,

            detail=(
                "ไม่พบ Whisper binary: "
                + WHISPER_BIN
            )

        )


    if not os.path.exists(
        WHISPER_MODEL
    ):

        raise HTTPException(

            status_code=500,

            detail=(
                "ไม่พบ Whisper model: "
                + WHISPER_MODEL
            )

        )


    if shutil.which(
        "ffmpeg"
    ) is None:

        raise HTTPException(

            status_code=500,

            detail="ไม่พบ FFmpeg"

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


            # =================================================
            # Save uploaded file
            # =================================================

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
            print("UPLOAD START")
            print("=" * 60)

            print(
                "Filename:",
                file.filename
            )

            print(
                "Input size:",
                input_size,
                "bytes"
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

            print(
                " ".join(ffmpeg_cmd)
            )


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

                print(
                    "FFMPEG ERROR:"
                )

                print(
                    ffmpeg_result.stderr[-5000:]
                )


                raise HTTPException(

                    status_code=500,

                    detail=(

                        "FFmpeg ไม่สามารถ "
                        "แปลงเสียงได้\n\n"

                        + ffmpeg_result.stderr[-3000:]

                    )

                )


            # =================================================
            # Audio exists
            # =================================================

            if not audio_file.exists():

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "FFmpeg ไม่ได้สร้าง audio.wav"
                    )

                )


            audio_size = (
                audio_file.stat().st_size
            )


            print(
                "AUDIO EXISTS:",
                True
            )

            print(
                "AUDIO SIZE:",
                audio_size
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

                    detail=(
                        "WAV ต้องเป็น Mono"
                    )

                )


            if wav_info["sample_rate"] != 16000:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "WAV ต้องเป็น 16000 Hz"
                    )

                )


            if wav_info["sample_width"] != 2:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "WAV ต้องเป็น 16-bit"
                    )

                )


            # =================================================
            # Whisper
            # =================================================

            print("=" * 60)
            print("WHISPER START")
            print("=" * 60)


            print(
                "WHISPER BIN:",
                WHISPER_BIN
            )

            print(
                "WHISPER MODEL:",
                WHISPER_MODEL
            )

            print(
                "WHISPER BIN EXISTS:",
                os.path.exists(
                    WHISPER_BIN
                )
            )

            print(
                "WHISPER MODEL EXISTS:",
                os.path.exists(
                    WHISPER_MODEL
                )
            )


            # =================================================
            # Whisper command
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


            print(
                "WHISPER COMMAND:"
            )

            print(
                " ".join(whisper_cmd)
            )


            # =================================================
            # Run Whisper
            # =================================================

            whisper_result = subprocess.run(

                whisper_cmd,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True,

                timeout=300

            )


            print("=" * 60)
            print("WHISPER RESULT")
            print("=" * 60)


            print(
                "RETURN CODE:",
                whisper_result.returncode
            )


            print(
                "STDOUT:"
            )

            print(
                whisper_result.stdout[-10000:]
            )


            print(
                "STDERR:"
            )

            print(
                whisper_result.stderr[-10000:]
            )


            # =================================================
            # Whisper failed
            # =================================================

            if whisper_result.returncode != 0:

                error_message = (

                    "Whisper ทำงานไม่สำเร็จ\n\n"

                    f"Return code: "
                    f"{whisper_result.returncode}\n\n"

                    "STDOUT:\n"

                    f"{whisper_result.stdout[-4000:]}\n\n"

                    "STDERR:\n"

                    f"{whisper_result.stderr[-4000:]}"

                )


                raise HTTPException(

                    status_code=500,

                    detail=error_message

                )


            # =================================================
            # Get transcript
            # =================================================

            transcript = (

                whisper_result.stdout
                .strip()

            )


            if not transcript:

                transcript = (
                    "ไม่พบข้อความเสียงพูด"
                )


            # =================================================
            # Clean Whisper log lines
            # =================================================

            lines = []


            for line in transcript.splitlines():

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


                lines.append(line)


            clean_transcript = (
                "\n".join(lines)
                .strip()
            )


            if not clean_transcript:

                clean_transcript = transcript


            # =================================================
            # Complete
            # =================================================

            print("=" * 60)
            print("TRANSCRIPTION COMPLETE")
            print("=" * 60)

            print(
                clean_transcript
            )


            # =================================================
            # Response
            # =================================================

            return {

                "status":
                    "ok",

                "version":
                    "1.8",

                "filename":
                    file.filename,

                "transcript":
                    clean_transcript,

                "audio": {

                    "sample_rate":
                        wav_info[
                            "sample_rate"
                        ],

                    "channels":
                        wav_info[
                            "channels"
                        ],

                    "duration_seconds":
                        round(

                            wav_info[
                                "frames"
                            ]

                            /

                            wav_info[
                                "sample_rate"
                            ],

                            2

                        )

                }

            }


    # =====================================================
    # HTTPException
    # =====================================================

    except HTTPException:

        raise


    # =====================================================
    # Timeout
    # =====================================================

    except subprocess.TimeoutExpired:

        raise HTTPException(

            status_code=500,

            detail=(

                "การประมวลผลใช้เวลานานเกินไป "
                "กรุณาลองวิดีโอที่สั้นลง"

            )

        )


    # =====================================================
    # Unexpected error
    # =====================================================

    except Exception as e:

        print(
            "=" * 60
        )

        print(
            "UNEXPECTED ERROR:"
        )

        print(
            repr(e)
        )

        print(
            "=" * 60
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "เกิดข้อผิดพลาด:\n"
                + repr(e)
            )

        )
