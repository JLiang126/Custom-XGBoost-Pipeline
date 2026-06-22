FROM python:3.10-slim AS builder

# Install the C++ compilers and Cmake needed for pybind11 and remove the previous installs
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /app

COPY . .

RUN pip install --upgrade pip setuptools wheel pybind11

RUN pip install --no-build-isolation .

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY . .

WORKDIR /app/experiments

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "train.py"]