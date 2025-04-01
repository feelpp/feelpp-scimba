FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-24.04


# include mpi+python3+pip3
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    cmake g++ openmpi-bin libopenmpi-dev python3 python3-pip python3-dev \
    libboost-test-dev libboost-serialization-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
