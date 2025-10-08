FROM python:3.10-slim 

RUN apt-get update && apt-get install -y \ 
    libgl1 \ 
    libglib2.0-0 \ 
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /app 

COPY requirements.txt . 

RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt 

COPY . . 

EXPOSE 5001 

# Command to run the application using Gunicorn 
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "2", "run:app"]