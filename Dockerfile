FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    DB_PATH=/app/chess_atlas.db \
    DB_URL=https://github.com/JamesVong/ChessAtlasBackend/releases/download/v1.0.0/chess_atlas.db

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

COPY . .

EXPOSE 5001

CMD [ "sh", "-c", "[ ! -f \"$DB_PATH\" ] && wget -q -O \"$DB_PATH\" \"$DB_URL\"; exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --max-requests 40 --max-requests-jitter 10 --timeout 120 run:app" ]
