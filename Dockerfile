FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --max-requests 40 --max-requests-jitter 10 --timeout 120 run:app
