from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='../env/model.env')
MODEL_PATH = os.getenv('MODEL_PATH')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH')
AUTHORIZED_TOKENS = os.getenv("AUTHORIZED_TOKENS", "default_token").split(",")

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)

app = FastAPI()


class TextRequest(BaseModel):
    prompt: str
    min_length: int = 25
    max_length: int = 150
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95


@app.post("/generate/")
def generate_text(request: TextRequest, token: str = Header(None)):
    if token not in AUTHORIZED_TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized token - Please contact me @ clovis.lechien@epita.fr")

    input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        min_length=request.min_length,
        max_length=request.max_length,
        num_return_sequences=1,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        do_sample=True
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
