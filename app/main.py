from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Thai Dub V1.5")

# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def home():
    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return {
            "status": "ok",
            "message": "AI Thai Dub V1.5 is running"
        }

    return FileResponse(index_file)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "AI Thai Dub V1.5"
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
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

    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)

    output_file = upload_dir / file.filename

    contents = await file.read()

    with open(output_file, "wb") as f:
        f.write(contents)

    return {
        "status": "success",
        "filename": file.filename,
        "size": len(contents),
        "message": "อัปโหลดไฟล์สำเร็จ"
    }


@app.get("/api/config")
async def config():
    return {
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "service": "AI Thai Dub V1.5"
    }
