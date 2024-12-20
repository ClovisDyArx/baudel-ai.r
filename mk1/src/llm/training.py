from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from mk1.src.dataset.dataset import DatasetPoem, PoemsDataset
import wandb
from dotenv import load_dotenv
import os
from mk1.src.variables import MODEL_PATH, TOKENIZER_PATH

load_dotenv(dotenv_path='env/wandb.env')
wandb.login(key=os.getenv('WANDB_API_KEY'))
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)

dataset = DatasetPoem()
tokenized_data = dataset.tokenize(tokenizer)
poems_dataset = PoemsDataset(tokenized_data)

training_args = TrainingArguments(
    output_dir="results",            # Directory to save model checkpoints
    num_train_epochs=2,                # Number of epochs
    per_device_train_batch_size=1,     # Batch size per device
    per_device_eval_batch_size=1,      # Evaluation batch size
    logging_dir="logs",              # Directory for logs
    logging_steps=10,                  # Log every 10 steps
    save_steps=2000,                   # Save checkpoint every 2000 steps
    eval_strategy="steps",             # Evaluate at every step
    eval_steps=100,                    # Evaluate every 100 steps
    report_to="wandb",                 # Use WandB for logging
    learning_rate=5e-5,                # Learning rate
    warmup_steps=50,                   # Warmup steps
    weight_decay=0.01,                 # Weight decay
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=poems_dataset,
    eval_dataset=poems_dataset,
)

trainer.train()
