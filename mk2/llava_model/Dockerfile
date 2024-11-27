FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y curl python3 python3-pip git

RUN curl -fsSL https://ollama.com/install.sh | sh

EXPOSE 11434

ENV OLLAMA_HOST=0.0.0.0:11434

CMD bash -c "ollama serve & sleep 5 && \
            ollama pull llava:13b && \
            ollama run llava:13b && \
            tail -f /dev/null"
