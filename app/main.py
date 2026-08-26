from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.responses import (
    FileResponse,
    PlainTextResponse
)

from fastapi.staticfiles import StaticFiles

import os
import subprocess
import tempfile
import shutil
import wave
import re


# =========================================================
# AI Thai Dub V1.9
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"


WHISPER_BIN = (
    "/opt/whisper.cpp/build/bin/whisper-cli"
)

WHISPER_MODEL = (
    "/opt/whisper.cpp/models/ggml-tiny.bin"
)


app = FastAPI(
    title="AI Thai Dub V1.9"
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


# =========================================================
# Home
# =========================================================

@app.get("/")
def home():

    index_file = (
        STATIC_DIR / "index.html"
    )

    if index_file.exists():

        return FileResponse(
            index_file
        )

    return {
        "status": "ok",
        "service": "AI Thai Dub V1.9"
    }


# =========================================================
# Health
# =========================================================

@app.get("/api/health")
def health():

    return {

        "status": "ok",

        "service":
            "AI Thai Dub V1.9",

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
            ) is not None,

        "static_exists":
            STATIC_DIR.exists(),

        "index_exists":
            (
                STATIC_DIR /
                "index.html"
            ).exists()
    }


# =========================================================
# WAV information
# =========================================================

def get_wav_info(
    wav_file
):

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

        duration = (
            frames / sample_rate
            if sample_rate
            else 0
        )

        return {

            "channels":
                channels,

            "sample_width":
                sample_width,

            "sample_rate":
                sample_rate,

            "frames":
                frames,

            "duration_seconds":
                round(
                    duration,
                    2
                )
        }

    except Exception as e:

        raise RuntimeError(
            "ไม่สามารถอ่าน WAV: "
            + str(e)
        )


# =========================================================
# Clean Whisper Text
# =========================================================

def clean_transcript(
    text
):

    if not text:

        return ""

    lines = []

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # Whisper system logs
        if line.startswith(
            "whisper_"
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
            "main:"
        ):
            continue

        if line.startswith(
            "output_"
        ):
            continue

        lines.append(line)


    text = "\n".join(
        lines
    ).strip()


    # -----------------------------------------------------
    # Remove duplicated blank lines
    # -----------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# =========================================================
# Transcribe
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


    suffix = (
        Path(
            file.filename
        ).suffix.lower()
    )


    allowed_extensions = {

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


    if suffix not in allowed_extensions:

        raise HTTPException(

            status_code=400,

            detail=(
                "ไฟล์ประเภทนี้ยังไม่รองรับ\n\n"
                "รองรับ:\n"
                "MP4, MOV, M4V, AVI, MKV, "
                "WEBM, MP3, WAV, M4A, "
                "AAC, FLAC และ OGG"
            )
        )


    # =====================================================
    # Check dependencies
    # =====================================================

    if not os.path.exists(
        WHISPER_BIN
    ):

        raise HTTPException(

            status_code=500,

            detail=(
                "ไม่พบ Whisper binary\n\n"
                + WHISPER_BIN
            )
        )


    if not os.path.exists(
        WHISPER_MODEL
    ):

        raise HTTPException(

            status_code=500,

            detail=(
                "ไม่พบ Whisper model\n\n"
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
    # Temporary directory
    # =====================================================

    try:

        with tempfile.TemporaryDirectory() as temp:

            temp_dir = Path(temp)


            input_file = (
                temp_dir /
                f"input{suffix}"
            )


            audio_file = (
                temp_dir /
                "audio.wav"
            )


            # =================================================
            # Save upload
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

                    output.write(
                        chunk
                    )


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

            ffmpeg_command = [

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
            print("FFMPEG")
            print("=" * 60)


            ffmpeg_result = subprocess.run(

                ffmpeg_command,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True,

                timeout=180
            )


            if (
                ffmpeg_result.returncode
                != 0
            ):

                print(
                    ffmpeg_result.stderr
                )

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "FFmpeg ทำงานไม่สำเร็จ\n\n"
                        + ffmpeg_result.stderr[
                            -4000:
                        ]
                    )
                )


            # =================================================
            # Validate audio
            # =================================================

            if not audio_file.exists():

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "FFmpeg ไม่ได้สร้าง audio.wav"
                    )
                )


            wav_info = get_wav_info(
                audio_file
            )


            print(
                "WAV:",
                wav_info
            )


            if wav_info[
                "channels"
            ] != 1:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "WAV ไม่ใช่ Mono"
                    )
                )


            if wav_info[
                "sample_rate"
            ] != 16000:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "WAV ไม่ใช่ 16000 Hz"
                    )
                )


            if wav_info[
                "sample_width"
            ] != 2:

                raise HTTPException(

                    status_code=500,

                    detail=(
                        "WAV ไม่ใช่ 16-bit"
                    )
                )


            # =================================================
            # Whisper
            # =================================================

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

                "-nt"
            ]


            print("=" * 60)
            print("WHISPER")
            print("=" * 60)

            print(
                " ".join(
                    whisper_command
                )
            )


            whisper_result = subprocess.run(

                whisper_command,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True,

                timeout=300
            )


            print(
                "Return code:",
                whisper_result.returncode
            )


            print(
                "STDOUT:",
                whisper_result.stdout[
                    -10000:
                ]
            )


            print(
                "STDERR:",
                whisper_result.stderr[
                    -10000:
                ]
            )


            # =================================================
            # Whisper error
            # =================================================

            if (
                whisper_result.returncode
                != 0
            ):

                raise HTTPException(

                    status_code=500,

                    detail=(

                        "Whisper ทำงานไม่สำเร็จ\n\n"

                        f"Return code: "
                        f"{whisper_result.returncode}\n\n"

                        "STDOUT:\n"

                        + whisper_result.stdout[
                            -3000:
                        ]

                        +

                        "\n\nSTDERR:\n"

                        + whisper_result.stderr[
                            -3000:
                        ]
                    )
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
                    "1.9",

                "filename":
                    file.filename,

                "file_size":
                    input_size,

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
                        wav_info[
                            "duration_seconds"
                        ]
                }
            }


    except HTTPException:

        raise


    except subprocess.TimeoutExpired:

        raise HTTPException(

            status_code=500,

            detail=(
                "การประมวลผลใช้เวลานานเกินไป\n\n"
                "ลองใช้วิดีโอที่สั้นลง"
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
# Download transcript
# =========================================================

@app.post(
    "/api/download-text"
)
async def download_text(
    file: UploadFile = File(...)
):

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail="ไม่มีข้อความ"
        )


    return PlainTextResponse(
        content.decode(
            "utf-8",
            errors="replace"
        ),

        media_type="text/plain"
    )
