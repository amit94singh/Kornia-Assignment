FROM anibali/pytorch:1.13.0-nocuda-ubuntu22.04
RUN sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install kornia kornia-rs kornia_moons opencv-python fastapi uvicorn python-multipart