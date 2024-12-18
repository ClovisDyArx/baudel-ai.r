FROM nvidia/cuda:12.0.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y curl python3 python3-pip git \
    nvidia-cuda-toolkit nvidia-container-toolkit

RUN curl -fsSL https://ollama.com/install.sh | sh

EXPOSE 11434

ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_USE_GPU=1


CMD bash -c "ollama serve & sleep 5 && \
            ollama pull llava:13b && \
            ollama run llava:13b && \
            tail -f /dev/null"
