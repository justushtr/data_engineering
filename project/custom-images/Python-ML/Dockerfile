FROM python:3.11-slim

WORKDIR /app

COPY project/custom-images/Python-ML/requirements.txt . 

RUN pip install -r requirements.txt

COPY project/ML-scripts/ .