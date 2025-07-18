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

# Copy the script and set ownership
COPY rismpi.py ./
COPY train.csv ./
COPY test.csv ./
RUN chown mpiuser:mpiuser rismpi.py
RUN chown mpiuser:mpiuser train.csv
RUN chown mpiuser:mpiuser test.csv

# Switch to the non-root user
USER mpiuser