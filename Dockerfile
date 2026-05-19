FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake python3-dev pybind11-dev python3-pip

WORKDIR /app

# Copies everying in dir
COPY . . 

RUN pip install .

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

COPY experiments/ ./experiments/

RUN pip install --no-cache-dir -r experiments/requirements.txt

ENTRYPOINT ["python", "experiments/train.py"]