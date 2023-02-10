FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY requirements.txt requirements.txt
    
RUN apt-get update -y \
    && python3 -m pip install -r requirements.txt