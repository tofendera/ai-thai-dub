# AI Thai Dub V1.5

## What this version does
Safari/iPhone -> HTTPS web app -> server -> OpenAI transcription -> Thai translation -> Thai TTS -> FFmpeg -> MP4 result.

## Deploy
The included Dockerfile is suitable for a container host such as Render, Railway, Fly.io or your own VPS.

Set environment variable:
OPENAI_API_KEY=your_key

Start command:
uvicorn app.main:app --host 0.0.0.0 --port 8000

FFmpeg is included in the Docker image.

## Important
Use only video/audio you have permission to process. This prototype does not bypass DRM or extract protected media from streaming services.
