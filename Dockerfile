FROM python:3.10-slim

WORKDIR /app

RUN apt update && \
    apt install -y espeak-ng ffmpeg && \
    apt clean

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py .

EXPOSE 3000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
