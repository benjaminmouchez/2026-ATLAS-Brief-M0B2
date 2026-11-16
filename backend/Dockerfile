FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY backend/requirements.txt ./requirements.txt

RUN uv pip install --no-cache --system -r requirements.txt

COPY backend ./backend
COPY models ./models
COPY data ./data
EXPOSE 8000

