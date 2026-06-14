FROM python:3.11-slim

# MediaPipe / OpenCV / 视频解码所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖以利用层缓存
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY . .

# Hugging Face Spaces 默认 7860; Render/Railway 会注入 PORT
ENV PORT=7860 \
    MPLBACKEND=Agg
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}"]
