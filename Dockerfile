FROM python:3.10-slim

# Install the C++ compilers and Cmake needed for pybind11 and remove the previous installs
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /app

COPY . .

RUN pip install --upgrade pip setuptools==69.5.1 wheel pybind11 && \
    pip install --no-cache-dir -r experiments/requirements.txt

RUN pip install --no-build-isolation .

WORKDIR /app/experiments

ENTRYPOINT ["python", "train.py"]