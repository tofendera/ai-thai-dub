from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import subprocess
import tempfile


# ==========================================
# PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

WHISPER_BIN = "/opt/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/opt/whisper.cpp/models/ggml-tiny-q5_1.bin"


# ==========================================
# APP
# ==========================================

app = FastAPI(title="AI Thai Dub V1.7.1")


# ==========================================
# STATIC WEBSITE
# ==========================================

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
        "service": "AI Thai Dub V1.7.1"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "AI Thai Dub V1.7.1",
        "whisper_exists": os.path.exists(WHISPER_BIN),
        "model_exists": os.path.exists(WHISPER_MODEL),
        "ffmpeg_exists": _command_exists("ffmpeg")
    }


def _command_exists(command):

    try:
        result = subprocess.run(
            ["which", command],
            capture_output=True,
            text=True
        )

        return result.returncode == 0

    except Exception:
        return False


# ==========================================
# TRANSCRIPTION
# ==========================================

@app.post("/api/transcribe")
@app.post("/api/upload")
async def transcribe(file: UploadFile = File(...)):

    # --------------------------------------
    # Check filename
    # --------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="ไม่ได้เลือกไฟล์"
        )


    # --------------------------------------
    # Check extension
    # --------------------------------------

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
            detail=(
                "รองรับไฟล์ MP4, MOV, M4V, AVI, MKV, "
                "WEBM, MP3, WAV และ M4A"
            )
        )


    # --------------------------------------
    # Check Whisper
    # --------------------------------------

    if not os.path.exists(WHISPER_BIN):

        raise HTTPException(
            status_code=500,
            detail="ไม่พบ Whisper binary"
        )


    # --------------------------------------
    # Check model
    # --------------------------------------

    if not os.path.exists(WHISPER_MODEL):

        raise HTTPException(
            status_code=500,
            detail="ไม่พบ Whisper model"
        )


    try:

        # ==================================
        # Temporary directory
        # ==================================

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            input_file = temp_dir / f"input{suffix}"

            audio_file = temp_dir / "audio.wav"


            # ==================================
            # Save uploaded file
            # ==================================

            print("====================================", flush=True)
            print("UPLOAD START", flush=True)
            print("Filename:", file.filename, flush=True)
            print("Input:", input_file, flush=True)
            print("====================================", flush=True)

            with open(input_file, "wb") as f:

                while True:

                    chunk = await file.read(1024 * 1024)

                    if not chunk:
                        break

                    f.write(chunk)


            print(
                "Uploaded file exists:",
                input_file.exists(),
                flush=True
            )

            print(
                "Uploaded file size:",
                input_file.stat().st_size if input_file.exists() else 0,
                flush=True
            )


            # ==================================
            # Convert to WAV
            # ==================================

            print("====================================", flush=True)
            print("FFMPEG START", flush=True)
            print("====================================", flush=True)

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


            print(
                "FFMPEG COMMAND:",
                " ".join(ffmpeg_cmd),
                flush=True
            )


            ffmpeg_result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )


            print(
                "FFMPEG RETURN CODE:",
                ffmpeg_result.returncode,
                flush=True
            )


            print(
                "FFMPEG STDOUT:",
                ffmpeg_result.stdout[-2000:],
                flush=True
            )


            print(
                "FFMPEG STDERR:",
                ffmpeg_result.stderr[-4000:],
                flush=True
            )


            print(
                "Audio exists:",
                audio_file.exists(),
                flush=True
            )


            if audio_file.exists():

                print(
                    "Audio size:",
                    audio_file.stat().st_size,
                    flush=True
                )


            # ==================================
            # FFmpeg error
            # ==================================

            if ffmpeg_result.returncode != 0:

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "FFmpeg failed\n\n"
                        "STDERR:\n"
                        + ffmpeg_result.stderr[-3000:]
                    )
                )


            if not audio_file.exists():

                raise HTTPException(
                    status_code=500,
                    detail="FFmpeg ไม่สร้างไฟล์ audio.wav"
                )


            # ==================================
            # Whisper
            # ==================================

            print("====================================", flush=True)
            print("WHISPER START", flush=True)
            print("====================================", flush=True)


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


            print(
                "WHISPER BIN:",
                WHISPER_BIN,
                flush=True
            )


            print(
                "WHISPER MODEL:",
                WHISPER_MODEL,
                flush=True
            )


            print(
                "WHISPER BIN EXISTS:",
                os.path.exists(WHISPER_BIN),
                flush=True
            )


            print(
                "WHISPER MODEL EXISTS:",
                os.path.exists(WHISPER_MODEL),
                flush=True
            )


            print(
                "AUDIO EXISTS:",
                audio_file.exists(),
                flush=True
            )


            print(
                "WHISPER COMMAND:",
                " ".join(whisper_cmd),
                flush=True
            )


            # ==================================
            # Run Whisper
            # ==================================

            whisper_result = subprocess.run(
                whisper_cmd,
                capture_output=True,
                text=True
            )


            # ==================================
            # Whisper Debug
            # ==================================

            print(
                "WHISPER RETURN CODE:",
                whisper_result.returncode,
                flush=True
            )


            print(
                "WHISPER STDOUT:",
                whisper_result.stdout[-4000:],
                flush=True
            )


            print(
                "WHISPER STDERR:",
                whisper_result.stderr[-4000:],
                flush=True
            )


            print("====================================", flush=True)
            print("WHISPER END", flush=True)
            print("====================================", flush=True)


            # ==================================
            # Whisper Error
            # ==================================

            if whisper_result.returncode != 0:

                stdout_debug = whisper_result.stdout[-2000:]

                stderr_debug = whisper_result.stderr[-4000:]


                if not stdout_debug:
                    stdout_debug = "(empty)"


                if not stderr_debug:
                    stderr_debug = "(empty)"


                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Whisper transcription failed\n\n"
                        f"Return code: "
                        f"{whisper_result.returncode}\n\n"
                        f"STDOUT:\n"
                        f"{stdout_debug}\n\n"
                        f"STDERR:\n"
                        f"{stderr_debug}"
                    )
                )


            # ==================================
            # Get transcript
            # ==================================

            transcript = whisper_result.stdout.strip()


            if not transcript:

                transcript = (
                    "ไม่สามารถตรวจพบเสียงพูดในไฟล์ได้"
                )


            # ==================================
            # Success
            # ==================================

            print("====================================", flush=True)
            print("TRANSCRIPTION SUCCESS", flush=True)
            print("Transcript:", transcript, flush=True)
            print("====================================", flush=True)


            return {

                "status": "ok",

                "filename": file.filename,

                "transcript": transcript

            }


    # ======================================
    # HTTP Exception
    # ======================================

    except HTTPException:

        raise


    # ======================================
    # Unexpected Error
    # ======================================

    except Exception as e:

        print("====================================", flush=True)
        print("UNEXPECTED ERROR", flush=True)
        print(str(e), flush=True)
        print("====================================", flush=True)


        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาด: {str(e)}"
        )
