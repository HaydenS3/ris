FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY rismpi.py ./

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

# Create a working directory and set ownership
RUN mkdir -p /app && chown mpiuser:mpiuser /app

# Copy the script and set ownership
COPY rismpi.py /app/
RUN chown mpiuser:mpiuser /app/rismpi.py

# Switch to the non-root user
USER mpiuser

# Set working directory
WORKDIR /app