from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

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

APP_VERSION = "1.8"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

STATIC_DIR = PROJECT_DIR / "static"

WHISPER_BIN = Path(
    "/opt/whisper.cpp/build/bin/whisper-cli"
)

WHISPER_MODEL = Path(
    "/opt/whisper.cpp/models/ggml-tiny.bin"
)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="AI Thai Dub V1.8",
    version=APP_VERSION
)


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

    if index_file.exists():

        return FileResponse(
            str(index_file)
        )

    return {
        "status": "ok",
        "service": "AI Thai Dub V1.8",
        "message": "Static website not found"
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/api/health")
def health():

    ffmpeg_path = shutil.which("ffmpeg")

    return {

        "status": "ok",

        "service":
            "AI Thai Dub V1.8",

        "version":
            APP_VERSION,

        "whisper_exists":
            WHISPER_BIN.is_file(),

        "whisper_executable":
            os.access(
                WHISPER_BIN,
                os.X_OK
            ),

        "model_exists":
            WHISPER_MODEL.is_file(),

        "model_size":
            (
                WHISPER_MODEL.stat().st_size
                if WHISPER_MODEL.is_file()
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
            ).exists()
    }


# =========================================================
# Whisper diagnostic
# =========================================================

@app.get("/api/whisper-test")
def whisper_test():

    result = {

        "whisper_exists":
            WHISPER_BIN.is_file(),

        "model_exists":
            WHISPER_MODEL.is_file(),

        "whisper_path":
            str(WHISPER_BIN),

        "model_path":
            str(WHISPER_MODEL)
    }


    if not WHISPER_BIN.is_file():

        result["status"] = "error"

        result["error"] = (
            "ไม่พบ whisper-cli"
        )

        return result


    if not WHISPER_MODEL.is_file():

        result["status"] = "error"

        result["error"] = (
            "ไม่พบ Whisper model"
        )

        return result


    try:

        test_result = subprocess.run(

            [
                str(WHISPER_BIN),
                "--help"
            ],

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            text=True,

            timeout=30
        )


        result["status"] = "ok"

        result["return_code"] = (
            test_result.returncode
        )

        result["stdout"] = (
            test_result.stdout[-3000:]
        )

        result["stderr"] = (
            test_result.stderr[-3000:]
        )


        return result


    except Exception as e:

        result["status"] = "error"

        result["error"] = repr(e)

        return result


# =========================================================
# WAV validation
# =========================================================

def check_wav(wav_file: Path):

    try:

        with wave.open(
            str(wav_file),
            "rb"
        ) as wav:

            channels = (
                wav.getnchannels()
            )

            sample_width = (
                wav.getsampwidth()
            )

            sample_rate = (
                wav.getframerate()
            )

            frames = (
                wav.getnframes()
            )


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
            "WAV ตรวจสอบไม่ได้: "
            + str(e)
        )


# =========================================================
# FFmpeg conversion
# =========================================================

def convert_to_wav(
    input_file: Path,
    audio_file: Path
):

    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:

        raise RuntimeError(
            "ไม่พบ FFmpeg"
        )


    command = [

        ffmpeg,

        "-y",

        "-hide_banner",

        "-loglevel",
        "error",

        "-i",
        str(input_file),

        "-vn",

        "-ac",
        "1",

        "-ar",
        "16000",

        "-sample_fmt",
        "s16",

        "-c:a",
        "pcm_s16le",

        str(audio_file)
    ]


    print(
        "=" * 60
    )

    print(
        "FFMPEG START"
    )

    print(
        " ".join(command)
    )


    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,

        timeout=180
    )


    print(
        "FFMPEG RETURN CODE:",
        result.returncode
    )


    if result.stdout:

        print(
            "FFMPEG STDOUT:"
        )

        print(
            result.stdout[-3000:]
        )


    if result.stderr:

        print(
            "FFMPEG STDERR:"
        )

        print(
            result.stderr[-5000:]
        )


    if result.returncode != 0:

        raise RuntimeError(

            "FFmpeg ไม่สามารถแปลงเสียงได้\n\n"
            + result.stderr[-4000:]
        )


    if not audio_file.exists():

        raise RuntimeError(
            "FFmpeg ไม่ได้สร้าง audio.wav"
        )


    if audio_file.stat().st_size <= 0:

        raise RuntimeError(
            "audio.wav มีขนาด 0 bytes"
        )


    return result


# =========================================================
# Whisper transcription
# =========================================================

def run_whisper(
    audio_file: Path
):

    if not WHISPER_BIN.is_file():

        raise RuntimeError(
            "ไม่พบ Whisper binary:\n"
            + str(WHISPER_BIN)
        )


    if not WHISPER_MODEL.is_file():

        raise RuntimeError(
            "ไม่พบ Whisper model:\n"
            + str(WHISPER_MODEL)
        )


    # =====================================================
    # IMPORTANT
    #
    # -ng = disable GPU
    # -t 2 = use 2 CPU threads
    # -l th = Thai
    # -nt = no timestamps
    # =====================================================

    command = [

        str(WHISPER_BIN),

        "-m",
        str(WHISPER_MODEL),

        "-f",
        str(audio_file),

        "-l",
        "th",

        "-t",
        "2",

        "-ng",

        "-nt"
    ]


    print(
        "=" * 60
    )

    print(
        "WHISPER START"
    )

    print(
        "WHISPER BIN:",
        str(WHISPER_BIN)
    )

    print(
        "WHISPER MODEL:",
        str(WHISPER_MODEL)
    )

    print(
        "WHISPER COMMAND:"
    )

    print(
        " ".join(command)
    )


    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,

        timeout=300
    )


    print(
        "=" * 60
    )

    print(
        "WHISPER RESULT"
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


    return result


# =========================================================
# Clean Whisper output
# =========================================================

def clean_transcript(text: str):

    if not text:

        return ""


    lines = []


    for raw_line in text.splitlines():

        line = raw_line.strip()


        if not line:

            continue


        # -----------------------------------------------
        # Skip common Whisper / GGML diagnostic lines
        # -----------------------------------------------

        prefixes = (

            "whisper_",

            "system_info:",

            "ggml_",

            "main:",

            "encode_",

            "decode_",

            "loading model",

            "whisper_print"
        )


        if line.startswith(prefixes):

            continue


        lines.append(line)


    return "\n".join(
        lines
    ).strip()


# =========================================================
# Transcribe API
# =========================================================

@app.post("/api/transcribe")
@app.post("/api/upload")
async def transcribe(
    file: UploadFile = File(...)
):

    # =====================================================
    # File validation
    # =====================================================

    if not file.filename:

        raise HTTPException(

            status_code=400,

            detail="ไม่ได้เลือกไฟล์"
        )


    original_name = (
        Path(file.filename).name
    )


    suffix = (
        Path(original_name)
        .suffix
        .lower()
    )


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
    # Environment validation
    # =====================================================

    if not WHISPER_BIN.is_file():

        raise HTTPException(

            status_code=500,

            detail=(
                "ไม่พบ Whisper binary\n\n"
                + str(WHISPER_BIN)
            )
        )


    if not WHISPER_MODEL.is_file():

        raise HTTPException(

            status_code=500,

            detail=(
                "ไม่พบ Whisper model\n\n"
                + str(WHISPER_MODEL)
            )
        )


    if shutil.which("ffmpeg") is None:

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

            print(
                "=" * 60
            )

            print(
                "UPLOAD START"
            )

            print(
                "Filename:",
                original_name
            )


            total_bytes = 0


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


                    output.write(
                        chunk
                    )

                    total_bytes += (
                        len(chunk)
                    )


            if total_bytes <= 0:

                raise HTTPException(

                    status_code=400,

                    detail="ไฟล์ว่าง"
                )


            print(
                "Input size:",
                total_bytes,
                "bytes"
            )


            # =================================================
            # Convert to WAV
            # =================================================

            try:

                convert_to_wav(
                    input_file,
                    audio_file
                )

            except Exception as e:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "FFmpeg แปลงไฟล์ไม่สำเร็จ\n\n"
                        + str(e)
                    )
                )


            # =================================================
            # Validate WAV
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
            # Run Whisper
            # =================================================

            try:

                whisper_result = run_whisper(
                    audio_file
                )

            except subprocess.TimeoutExpired:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "Whisper ใช้เวลานานเกินไป "
                        "กรุณาลองวิดีโอที่สั้นลง"
                    )
                )


            except Exception as e:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "ไม่สามารถเริ่ม Whisper ได้\n\n"
                        + repr(e)
                    )
                )


            # =================================================
            # Whisper failed
            # =================================================

            if whisper_result.returncode != 0:

                stdout = (
                    whisper_result.stdout
                    or ""
                )

                stderr = (
                    whisper_result.stderr
                    or ""
                )


                detail = (

                    "Whisper ทำงานไม่สำเร็จ\n\n"

                    f"Return code: "
                    f"{whisper_result.returncode}\n\n"

                    "STDOUT:\n"
                    f"{stdout[-5000:]}\n\n"

                    "STDERR:\n"
                    f"{stderr[-5000:]}"
                )


                raise HTTPException(

                    status_code=500,

                    detail=detail
                )


            # =================================================
            # Clean transcript
            # =================================================

            transcript = clean_transcript(

                whisper_result.stdout
            )


            if not transcript:

                transcript = (
                    "ไม่พบข้อความเสียงพูด"
                )


            print(
                "=" * 60
            )

            print(
                "TRANSCRIPTION COMPLETE"
            )

            print(
                transcript
            )


            # =================================================
            # Duration
            # =================================================

            duration_seconds = round(

                wav_info["frames"]
                / wav_info["sample_rate"],

                2
            )


            # =================================================
            # Response
            # =================================================

            return {

                "status":
                    "ok",

                "version":
                    APP_VERSION,

                "filename":
                    original_name,

                "transcript":
                    transcript,

                "audio": {

                    "sample_rate":
                        wav_info[
                            "sample_rate"
                        ],

                    "channels":
                        wav_info[
                            "channels"
                        ],

                    "sample_width":
                        wav_info[
                            "sample_width"
                        ],

                    "duration_seconds":
                        duration_seconds
                }
            }


    except HTTPException:

        raise


    except Exception as e:

        print(
            "=" * 60
        )

        print(
            "UNEXPECTED ERROR"
        )

        print(
            repr(e)
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "เกิดข้อผิดพลาด:\n"
                + repr(e)
            )
        )
