FROM python:3.11.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt