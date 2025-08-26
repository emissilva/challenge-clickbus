FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tratar_dados.py .
COPY scripts/ scripts/
COPY data/ data/