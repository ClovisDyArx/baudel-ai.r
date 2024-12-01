# ***Backend***

Contains the backend for the application.
Written in Python using FastAPI.


## **Overview**
### [**backend.py**](backend.py)
- **Description**
  - The backend for the application.
  - Contains the following:
    - **Routes**:
      - `(POST) /generate`: The route for generating text using the LLaVA model.
      - `(POST) /tts`: The route for converting text to speech, using Google Text-to-Speech.
    - **Functions**:
      -  `validate_token`: Handles token validation, raising an exception if the token is invalid.
      - `preprocess_text_for_tts`: Preprocesses text for text-to-speech, removing unwanted characters.
    - **Classes**:
      - `PromptRequest`: The class dictating the structure of the request for the LLaVA model.

### [**Dockerfile**](Dockerfile)
- **Description**
  - The Dockerfile for the application.
  - Contains the following:
    - **Base Image**: `python:3.9-slim`
    - **Dependencies**: `requirements.txt`
    - **Environment Variables**:
      - `PORT`: `8000`

### [**requirements.txt**](requirements.txt)
- **Description**
  - The dependencies for the application.
  - Contains the following:
    - `python-dotenv`
    - `fastapi`
    - `pydantic`
    - `requests`
    - `uvicorn`
    - `gTTS`