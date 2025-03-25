FROM nvcr.io/nvidia/tensorrt:22.12-py3

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV QT_QPA_PLATFORM=offscreen

# Install all required APT dependencies in one layer
RUN apt update && apt install -y \
    android-tools-adb \
    android-tools-fastboot \
    udev \
    wget \
    git \
    curl \
    unzip \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrender-dev \
    libsm6 \
    libegl1 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libopencv-dev \
    libgtk2.0-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
	v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Update GCC and libstdc++ via conda
RUN conda install -n base -c conda-forge libstdcxx-ng

# Install mamba and create conda environment
WORKDIR /workspace
COPY environment.yml .
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -f environment.yml

# Set default environment
SHELL ["conda", "run", "-n", "new", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Start with interactive bash by default
CMD ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "new", "/bin/bash"]

