# FIP Real-time IMU Motion Renderer
# Default: CPU-only, osmesa software rendering (no GPU required)
# For GPU: see docker-compose.gpu.yml

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
# osmesa = pure software OpenGL, works without any GPU/display
ENV PYOPENGL_PLATFORM=osmesa

# System deps:
#   libosmesa6       – Mesa software OpenGL (pyrender headless rendering)
#   libglib2.0-0     – glib (trimesh/pyrender indirect dep)
#   libgomp1         – OpenMP (PyTorch parallel ops)
#   ffmpeg           – video encoding (imageio-ffmpeg backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# IMU data port (TCP)
EXPOSE 9000
# HTTP MJPEG stream port
EXPOSE 8080

CMD ["python", "stream_server.py"]
