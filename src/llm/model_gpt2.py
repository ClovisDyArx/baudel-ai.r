from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from src.dataset.dataset import DatasetPoem, PoemsDataset
import wandb
from dotenv import load_dotenv
import os

# Load environment variables for WandB
load_dotenv(dotenv_path='env/wandb.env')
wandb.login(key=os.getenv('WANDB_API_KEY'))

# Load GPT-2 model and tokenizer
model_name = "gpt2-medium"  # TODO: try "gpt2-large" or "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load and preprocess dataset
dataset = DatasetPoem()
tokenized_data = dataset.tokenize(tokenizer)

# Convert tokenized data into a PyTorch Dataset
poems_dataset = PoemsDataset(tokenized_data)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",            # Directory to save model checkpoints
    num_train_epochs=3,                # Number of epochs
    per_device_train_batch_size=1,     # Batch size per device
    per_device_eval_batch_size=1,      # Evaluation batch size
    logging_dir="./logs",              # Directory for logs
    logging_steps=10,                  # Log every 10 steps
    save_steps=500,                    # Save checkpoint every 500 steps
    eval_strategy="steps",       # Evaluate at every step
    report_to="wandb",                 # Use WandB for logging
    learning_rate=5e-5,                # Learning rate
    warmup_steps=50,                   # Warmup steps
    weight_decay=0.01,                 # Weight decay
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=poems_dataset,  # Pass the custom dataset
    eval_dataset=poems_dataset,   # Use the same dataset for evaluation (for simplicity)
)

# Start training
trainer.train()
