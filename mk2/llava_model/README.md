# ***LLaVA***: Large Language and Vision Assistant

The *LLaVA* model is a multimodal transformer model that can be used for a variety of tasks,
including image captioning, visual question answering, and image-text retrieval.
Here, we use it to generate love letters from images and personnalized messages.

## **Overview**
### [**Dockerfile**](Dockerfile)
- **Description**
  - The Dockerfile for the application.
  - Installs and runs ollama to serve the LLaVA model.
  - Contains the following:
    - **Base Image**: `ubuntu:20.04`
    - **Environment Variables**:
      - `PORT`: `11434`
      - `OLLAMA_HOST=0.0.0.0:11434`