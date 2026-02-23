# Dockerfile
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install torch-geometric
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    && pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    && pip install --no-cache-dir torch-geometric

# Copy source code
COPY . .

# Create directories for data and outputs
RUN mkdir -p data/tcia_nsclc data/tcga_luad outputs checkpoints

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "main.py"]
