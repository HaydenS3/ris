# Use NVIDIA's CUDA runtime image as base
FROM nvidia/cuda:12.9.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with GPU support
RUN pip3 install --upgrade pip && pip3 install \
    mpi4py \
    tqdm \
    numpy \
    pandas \
    tensorflow[and-cuda] \
    scikit-learn

# Create a non-root user
RUN groupadd -r mpiuser && useradd -r -g mpiuser mpiuser

RUN mkdir -p /app && chown mpiuser:mpiuser /app
RUN mkdir -p /app/store && chown mpiuser:mpiuser /app/store

# Copy the script and set ownership
COPY rismpigpu.py /app/
COPY train.csv /app/
COPY test.csv /app/
RUN chown mpiuser:mpiuser /app/rismpigpu.py
RUN chown mpiuser:mpiuser /app/train.csv
RUN chown mpiuser:mpiuser /app/test.csv

# Switch to the non-root user
USER mpiuser