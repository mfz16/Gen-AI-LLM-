import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from tensorboardX import SummaryWriter


# Define your custom callback for logging


# Define your custom callback for logging
from transformers import TrainerCallback
class CustomTensorBoardCallback(TrainerCallback):
    def __init__(self, writer):
        self.writer = writer

    def on_step(self, args, state, control, model, optimizer, step):
        # Log custom metrics using self.writer.add_scalar()
        self.writer.add_scalar("custom_metric", state.log_history[-1]["custom_metric"], step)



# Define a custom dataset class for training
class CustomTrainingDataset(Dataset):
    def __init__(self, tokenizer, num_samples=1000, sequence_length=20):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

        # Generate random sequences as training data
        self.data = [torch.randint(0, tokenizer.vocab_size, (self.sequence_length,)) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        attention_mask = torch.ones_like(input_ids)  # Dummy attention mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # For language modeling, labels are usually the same as input_ids
        }

class CustomEvaluationDataset(Dataset):
    def __init__(self, tokenizer, num_samples=200, sequence_length=20):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

        # Generate random sequences as evaluation data
        self.data = [torch.randint(0, tokenizer.vocab_size, (self.sequence_length,)) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        attention_mask = torch.ones_like(input_ids)  # Dummy attention mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # For language modeling, labels are usually the same as input_ids
        }

# Step 1: Load a pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 2: Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Create instances of the custom datasets
train_dataset = CustomTrainingDataset(tokenizer, num_samples=1000, sequence_length=20)
eval_dataset = CustomEvaluationDataset(tokenizer, num_samples=200, sequence_length=20)


# Set up TensorBoard logging
log_dir = "./gpt2hindi/results/logs"
writer = SummaryWriter(log_dir=log_dir)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2hindi/results/output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=log_dir,
    logging_steps=1000,
    save_steps=5000,
    evaluation_strategy='steps',
    eval_steps=5000,
    load_best_model_at_end=True,
    report_to='tensorboard',
)

# Create data loaders for training and evaluation
train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size)

# Create a Trainer with the TensorBoardXCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the DataLoader
    eval_dataset=eval_dataset,    # Use the DataLoader
    data_collator=None,
    callbacks=[CustomTensorBoardCallback(writer)]
)

# Train the model
trainer.train()

# Close the TensorBoard writer
writer.close()
