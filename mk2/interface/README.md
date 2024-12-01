# ***Interface***

Contains the interface for the application.
Written in Python using Streamlit.


## **Overview**
### [**app.py**](app.py)
- **Description**
  - The main interface for the application.
  - Uses Streamlit to create a web application.
  - Runs on port [8501](http://localhost:8501).
  - Contains the following features:
    - **MLLM - LLaVA**
    - **TTS - gTTS**
  - Supports 9 languages:
    - **English**
    - **Spanish**
    - **French**
    - **German**
    - **Japanese**
    - **Chinese**
    - **Thai**
    - **Russian**
    - **Hindi**

### [**Dockerfile**](Dockerfile)
- **Description**
  - The Dockerfile for the application.
  - Contains the following:
    - **Base Image**: `python:3.9-slim`
    - **Dependencies**: `requirements.txt`
    - **Environment Variables**:
      - `PORT`: `8501`

### [**requirements.txt**](requirements.txt)
- **Description**
  - The dependencies for the application.
  - Contains the following:
    - `streamlit`
    - `fastapi`
    - `pydantic`
    - `requests`
    - `uvicorn`

### [**translations.json**](translations.json)
- **Description**
  - The translations for the application.
  - Contains the following:
    - **Languages**:
      - **English**
      - **Spanish**
      - **French**
      - **German**
      - **Japanese**
      - **Chinese**
      - **Thai**
      - **Russian**
      - **Hindi**
  - **Note #1**: The translations are used for the TTS feature aswell as the interface.
  - **Note #2**: The translations are stored in a JSON format.
  - **Important Note**: The translations are AI generated and may not be accurate.