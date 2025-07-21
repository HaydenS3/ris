FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev

RUN pip3 install --upgrade pip && pip3 install \
    mpi4py \
    tqdm \
    numpy \
    pandas \
    tensorflow \
    keras \
    scikit-learn

# Create a non-root user
RUN groupadd -r mpiuser && useradd -r -g mpiuser mpiuser

RUN mkdir -p /app && chown mpiuser:mpiuser /app

# Copy the script and set ownership
COPY rismpigpu.py /app/
COPY train.csv /app/
COPY test.csv /app/
RUN chown mpiuser:mpiuser /app/rismpigpu.py
RUN chown mpiuser:mpiuser /app/train.csv
RUN chown mpiuser:mpiuser /app/test.csv

# Switch to the non-root user
USER mpiuser
