from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='env/model.env')
model_path = os.getenv('MODEL_PATH')

# model_name = "gpt2-medium"  # TODO: "gpt2-medium", "gpt2-large" or "gpt2-xl"

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

pre_prompt = ""  # "Write a love letter:\n"
main_prompt = "Dear Deborah,"

full_prompt = pre_prompt + main_prompt

input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    min_length=10,             # Minimum number of tokens in the generated text
    max_length=150,            # Maximum number of tokens in the generated text
    num_return_sequences=1,    # Number of sequences to generate
    temperature=0.7,           # Sampling temperature (higher = more random)
    top_k=50,                  # Consider top-k tokens for sampling
    top_p=0.95,                # Nucleus sampling (top-p tokens' cumulative probability >= 0.95)
    do_sample=True             # Enable sampling
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

if generated_text.startswith(pre_prompt):
    generated_text = generated_text[len(pre_prompt):]

print("Generated Text:")
print(generated_text)
