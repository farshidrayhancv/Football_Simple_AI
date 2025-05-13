# Dockerfile
FROM ubuntu:22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python pip and upgrade
RUN pip3 install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Install gdown for downloading files from Google Drive
RUN pip3 install gdown

# Create videos directory
RUN mkdir -p /app/videos

# Download sample videos
RUN gdown -O /app/videos/0bfacc_0.mp4 "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF" && \
    gdown -O /app/videos/2e57b9_0.mp4 "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf" && \
    gdown -O /app/videos/08fd33_0.mp4 "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-" && \
    gdown -O /app/videos/573e61_0.mp4 "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU" && \
    gdown -O /app/videos/121364_0.mp4 "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

# Create output directory
RUN mkdir -p /app/output

# Set the default command to bash
CMD ["/bin/bash"]
