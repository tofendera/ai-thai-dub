from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os
import shutil
import subprocess
import tempfile
import wave


# =========================================================
# AI Thai Dub V2.1
# Stable transcription version
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# ---------------------------------------------------------
# Whisper
# ---------------------------------------------------------

WHISPER_BIN = os.getenv(
    "WHISPER_BIN",
    "/opt/whisper.cpp/build/bin/whisper-cli"
)

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "/opt/whisper.cpp/models/ggml-tiny.bin"
)

# Maximum video duration allowed
MAX_DURATION_SECONDS = int(
    os.getenv("MAX_DURATION_SECONDS", "180")
)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="AI Thai Dub V2.1",
    version="2.1"
)


# =========================================================
# Static Website
# =========================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=STATIC_DIR
        ),
        name="static"
    )


@app.get("/")
def home():

    index_file = STATIC_DIR / "index.html"

    if index_file.exists():

        return FileResponse(
            index_file
        )

    return {
        "status": "ok",
        "service": "AI Thai Dub V2.1"
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/api/health")
def health():

    ffmpeg_path = shutil.which(
        "ffmpeg"
    )

    ffprobe_path = shutil.which(
        "ffprobe"
    )

    return {

        "status": "ok",

        "service":
            "AI Thai Dub V2.1",

        "whisper_exists":
            os.path.exists(
                WHISPER_BIN
            ),

        "model_exists":
            os.path.exists(
                WHISPER_MODEL
            ),

        "ffmpeg_exists":
            ffmpeg_path is not None,

        "ffprobe_exists":
            ffprobe_path is not None,

        "whisper_path":
            WHISPER_BIN,

        "model_path":
            WHISPER_MODEL,

        "max_duration_seconds":
            MAX_DURATION_SECONDS
    }


# =========================================================
# Get Media Duration
# =========================================================

def get_duration(media_file: Path) -> float:

    ffprobe = shutil.which(
        "ffprobe"
    )

    if not ffprobe:

        raise RuntimeError(
            "ไม่พบ ffprobe"
        )

    command = [

        ffprobe,

        "-v",
        "error",

        "-show_entries",
        "format=duration",

        "-of",
        "default=noprint_wrappers=1:nokey=1",

        str(media_file)
    ]

    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,

        timeout=30
    )

    if result.returncode != 0:

        raise RuntimeError(
            "ไม่สามารถอ่านความยาววิดีโอได้\n"
            + result.stderr[-2000:]
        )

    try:

        duration = float(
            result.stdout.strip()
        )

    except Exception:

        raise RuntimeError(
            "ไม่สามารถอ่านค่าความยาววิดีโอได้"
        )

    return duration


# =========================================================
# WAV Validation
# =========================================================

def check_wav(wav_file: Path):

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
            "ตรวจสอบ WAV ไม่ได้: "
            + str(e)
        )


# =========================================================
# Clean Whisper Output
# =========================================================

def clean_whisper_output(text: str):

    if not text:

        return ""

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Whisper system/log lines
        skip_prefixes = (

            "whisper_",

            "main:",

            "system_info:",

            "ggml_",

            "whisper_init",

            "model_load",

            "loading model",

            "print_timings"
        )

        if line.startswith(
            skip_prefixes
        ):
            continue

        lines.append(
            line
        )

    return "\n".join(
        lines
    ).strip()


# =========================================================
# Transcribe
# =========================================================

@app.post("/api/transcribe")
@app.post("/api/upload")
async def transcribe(
    file: UploadFile = File(...)
):

    # =====================================================
    # Check filename
    # =====================================================

    if not file.filename:

        raise HTTPException(

            status_code=400,

            detail=
                "ไม่ได้เลือกไฟล์"
        )


    suffix = Path(
        file.filename
    ).suffix.lower()


    # =====================================================
    # Allowed files
    # =====================================================

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
                "รองรับไฟล์ "
                "MP4, MOV, M4V, AVI, MKV, "
                "WEBM, MP3, WAV, M4A, "
                "AAC, FLAC และ OGG"
            )
        )


    # =====================================================
    # Check Whisper
    # =====================================================

    if not os.path.exists(
        WHISPER_BIN
    ):

        raise HTTPException(

            status_code=500,

            detail=(
                "ไม่พบ Whisper binary\n"
                + WHISPER_BIN
            )
        )


    if not os.path.exists(
        WHISPER_MODEL
    ):

        raise HTTPException(

            status_code=500,

            detail=(
                "ไม่พบ Whisper model\n"
                + WHISPER_MODEL
            )
        )


    # =====================================================
    # Check FFmpeg
    # =====================================================

    ffmpeg = shutil.which(
        "ffmpeg"
    )

    if not ffmpeg:

        raise HTTPException(

            status_code=500,

            detail=
                "ไม่พบ FFmpeg"
        )


    # =====================================================
    # Check FFprobe
    # =====================================================

    ffprobe = shutil.which(
        "ffprobe"
    )

    if not ffprobe:

        raise HTTPException(

            status_code=500,

            detail=
                "ไม่พบ FFprobe"
        )


    # =====================================================
    # Temporary Workspace
    # =====================================================

    try:

        with tempfile.TemporaryDirectory() as temp:

            temp_dir = Path(
                temp
            )


            input_file = (
                temp_dir
                / f"input{suffix}"
            )


            audio_file = (
                temp_dir
                / "audio.wav"
            )


            # =================================================
            # Save Upload
            # =================================================

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

                    total_bytes += len(
                        chunk
                    )


            if total_bytes <= 0:

                raise HTTPException(

                    status_code=400,

                    detail=
                        "ไฟล์ว่าง"
                )


            print("=" * 60)

            print(
                "AI THAI DUB V2.1"
            )

            print(
                "UPLOAD COMPLETE"
            )

            print(
                "Filename:",
                file.filename
            )

            print(
                "Size:",
                total_bytes,
                "bytes"
            )


            # =================================================
            # Get Duration
            # =================================================

            try:

                duration = get_duration(
                    input_file
                )

            except Exception as e:

                raise HTTPException(

                    status_code=400,

                    detail=
                        "อ่านความยาวไฟล์ไม่ได้\n"
                        + str(e)
                )


            print(
                "DURATION:",
                duration,
                "seconds"
            )


            # =================================================
            # Duration Check
            # =================================================

            if duration <= 0:

                raise HTTPException(

                    status_code=400,

                    detail=
                        "ไม่สามารถอ่านความยาววิดีโอได้"
                )


            if duration > MAX_DURATION_SECONDS:

                raise HTTPException(

                    status_code=400,

                    detail=(
                        "วิดีโอยาวเกินไป\n\n"
                        f"ความยาว: "
                        f"{round(duration, 1)} วินาที\n"
                        f"รองรับสูงสุด: "
                        f"{MAX_DURATION_SECONDS} วินาที\n\n"
                        "กรุณาลองวิดีโอที่สั้นลง"
                    )
                )


            # =================================================
            # FFmpeg
            # =================================================

            print("=" * 60)

            print(
                "FFMPEG START"
            )


            ffmpeg_command = [

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

                "-c:a",
                "pcm_s16le",

                str(audio_file)
            ]


            ffmpeg_result = subprocess.run(

                ffmpeg_command,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True,

                timeout=90
            )


            print(
                "FFMPEG RETURN:",
                ffmpeg_result.returncode
            )


            if ffmpeg_result.returncode != 0:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "FFmpeg ไม่สามารถแยกเสียงได้\n\n"
                        + ffmpeg_result.stderr[-3000:]
                    )
                )


            if not audio_file.exists():

                raise HTTPException(

                    status_code=500,

                    detail=
                        "FFmpeg ไม่ได้สร้าง audio.wav"
                )


            audio_size = (
                audio_file.stat().st_size
            )


            print(
                "AUDIO SIZE:",
                audio_size
            )


            # =================================================
            # WAV Check
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

                    detail=
                        "WAV ต้องเป็น Mono"
                )


            if wav_info["sample_rate"] != 16000:

                raise HTTPException(

                    status_code=500,

                    detail=
                        "WAV ต้องเป็น 16000 Hz"
                )


            if wav_info["sample_width"] != 2:

                raise HTTPException(

                    status_code=500,

                    detail=
                        "WAV ต้องเป็น 16-bit"
                )


            # =================================================
            # Whisper
            # =================================================

            print("=" * 60)

            print(
                "WHISPER START"
            )


            whisper_command = [

                WHISPER_BIN,

                "-m",
                WHISPER_MODEL,

                "-f",
                str(audio_file),

                "-l",
                "th",

                "-t",
                "2",

                "-nt",

                "-np"
            ]


            print(
                "WHISPER COMMAND:"
            )

            print(
                " ".join(
                    whisper_command
                )
            )


            # -------------------------------------------------
            # Timeout
            #
            # เผื่อเวลาให้ Whisper มากกว่าความยาววิดีโอ
            # -------------------------------------------------

            whisper_timeout = max(

                60,

                min(

                    240,

                    int(
                        duration * 4
                    ) + 30
                )
            )


            print(
                "WHISPER TIMEOUT:",
                whisper_timeout,
                "seconds"
            )


            try:

                whisper_result = subprocess.run(

                    whisper_command,

                    stdout=subprocess.PIPE,

                    stderr=subprocess.PIPE,

                    text=True,

                    timeout=whisper_timeout
                )

            except subprocess.TimeoutExpired:

                raise HTTPException(

                    status_code=504,

                    detail=(
                        "Whisper ใช้เวลานานเกินไป\n\n"
                        f"วิดีโอ: "
                        f"{round(duration, 1)} วินาที\n"
                        f"เวลาประมวลผลสูงสุด: "
                        f"{whisper_timeout} วินาที\n\n"
                        "กรุณาลองวิดีโอที่สั้นลง"
                    )
                )


            # =================================================
            # Whisper Result
            # =================================================

            print("=" * 60)

            print(
                "WHISPER RETURN:",
                whisper_result.returncode
            )


            print(
                "WHISPER STDOUT:"
            )

            print(
                whisper_result.stdout[-10000:]
            )


            print(
                "WHISPER STDERR:"
            )

            print(
                whisper_result.stderr[-10000:]
            )


            # =================================================
            # Whisper Error
            # =================================================

            if whisper_result.returncode != 0:

                error_text = (
                    whisper_result.stderr.strip()
                    or
                    whisper_result.stdout.strip()
                    or
                    "ไม่ทราบสาเหตุ"
                )


                raise HTTPException(

                    status_code=500,

                    detail=(
                        "Whisper ทำงานไม่สำเร็จ\n\n"
                        f"Return code: "
                        f"{whisper_result.returncode}\n\n"
                        f"{error_text[-4000:]}"
                    )
                )


            # =================================================
            # Clean Transcript
            # =================================================

            transcript = clean_whisper_output(

                whisper_result.stdout
            )


            if not transcript:

                transcript = (
                    "ไม่พบข้อความเสียงพูด"
                )


            # =================================================
            # Final Result
            # =================================================

            print("=" * 60)

            print(
                "TRANSCRIPTION COMPLETE"
            )

            print(
                transcript
            )


            return {

                "status":
                    "ok",

                "version":
                    "2.1",

                "filename":
                    file.filename,

                "duration_seconds":
                    round(
                        duration,
                        2
                    ),

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

                    "duration_seconds":
                        round(
                            wav_info["frames"]
                            /
                            wav_info["sample_rate"],
                            2
                        )
                }
            }


    # =====================================================
    # Timeout
    # =====================================================

    except subprocess.TimeoutExpired:

        raise HTTPException(

            status_code=504,

            detail=(
                "การประมวลผลใช้เวลานานเกินไป\n\n"
                "กรุณาลองวิดีโอที่สั้นลง"
            )
        )


    # =====================================================
    # HTTP Error
    # =====================================================

    except HTTPException:

        raise


    # =====================================================
    # Unexpected Error
    # =====================================================

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
                "เกิดข้อผิดพลาดในระบบ\n\n"
                + repr(e)
            )
        )
