FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY mpitest.py ./

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev

RUN pip3 install --upgrade pip && pip3 install \
    mpi4py \
    tqdm \
    numpy

# Create a non-root user
RUN groupadd -r mpiuser && useradd -r -g mpiuser mpiuser

# Create a working directory and set ownership
RUN mkdir -p /app && chown mpiuser:mpiuser /app

# Create mount point for external storage
RUN mkdir -p /app/store && chown mpiuser:mpiuser /app/store

# Copy the script and set ownership
COPY mpitest.py /app/
RUN chown mpiuser:mpiuser /app/mpitest.py /app/store

# Switch to the non-root user
USER mpiuser