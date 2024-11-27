from transformers import GPT2LMHeadModel, GPT2Tokenizer
from mk1.src.variables import MODEL_PATH, TOKENIZER_PATH

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)

prompt = "Dear Deborah, your "

input_ids = tokenizer.encode(prompt, return_tensors="pt")

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

print("Generated Text:")
print(generated_text)
