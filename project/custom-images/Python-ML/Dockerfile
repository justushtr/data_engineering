FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY project/custom-images/Python-ML/requirements.txt . 

RUN pip install -r requirements.txt

COPY project/ML-scripts/ .

CMD ["python", "-u", "minio_watcher.py"]