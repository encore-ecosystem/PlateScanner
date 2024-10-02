FROM python:3.12-alpine AS base
WORKDIR /app
RUN pip install poetry
COPY . .
RUN poetry install
ENTRYPOINT poetry run python main.py
