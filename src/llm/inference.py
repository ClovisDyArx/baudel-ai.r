from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./finetuned_models/gpt2-medium-500poems"
model_name = "gpt2-medium"  # TODO: "gpt2-medium", "gpt2-large" or "gpt2-xl"

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define a prompt for the model
prompt = "Dear Deborah, your smile is as bright as the sun and your "

input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    min_length=50,             # Minimum number of tokens in the generated text
    max_length=100,            # Maximum number of tokens in the generated text
    num_return_sequences=1,    # Number of sequences to generate
    temperature=0.5,           # Sampling temperature (higher = more random)
    top_k=50,                  # Consider top-k tokens for sampling
    top_p=0.95,                # Nucleus sampling (top-p tokens' cumulative probability >= 0.95)
    do_sample=True             # Enable sampling
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
