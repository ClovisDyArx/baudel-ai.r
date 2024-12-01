from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
from dotenv import load_dotenv
import os
from fastapi.responses import FileResponse
import re
from gtts import gTTS


LLAVA_URL = "http://llava_model:11434/api/generate"

load_dotenv(dotenv_path='.env')
SECRET_TOKEN = os.getenv("SECRET_TOKEN")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptRequest(BaseModel):
    model: str
    prompt: str
    images: list[str] = None


def validate_token(request: Request):
    token = request.headers.get("Authorization")

    if not token or token != f"Bearer {SECRET_TOKEN}":
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing token. Please provide the correct token."
        )


def preprocess_text_for_tts(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.post("/generate")
async def generate_love_letter(request: PromptRequest, token: str = Depends(validate_token)):
    logger.info(f"Received request: {request}")

    llava_payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False
    }

    if request.images:
        llava_payload["images"] = request.images

    try:
        response = requests.post(LLAVA_URL, json=llava_payload)
        logger.info(f"Response received: {response.status_code}")

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to generate text.")

        response_data = response.json()
        text = response_data.get("response", "")

        if not text:
            raise HTTPException(status_code=500, detail="No text generated.")

        return {"text": text}

    except requests.exceptions.RequestException as e:
        logger.error(f"Request to LLaVA model failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to LLaVA model: {e}")


@app.post("/tts")
async def text_to_speech(request: dict):
    text = request.get("text")
    text = preprocess_text_for_tts(text)
    language = request.get("language", "en")

    logger.info(f"TTS - Received text: {text}")

    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    try:
        tts = gTTS(text, lang=language)
        audio_file = "output.mp3"
        tts.save(audio_file)

        return FileResponse(audio_file, media_type="audio/mp3", filename="output.mp3")

    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {e}")
