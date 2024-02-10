import torch
from torch.utils.data import random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, \
    DataCollatorForLanguageModeling

# Step 1: Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Step 2: Tokenize the training data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Step 3: Prepare the training data
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="d:/test/chat/clean_chat_short.csv",
    block_size=128
)
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")

# Step 4: Create a TrainingArguments object
training_args = TrainingArguments(
    output_dir='./llmchat/results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./llmchat/logs',
    logging_steps=1000,
    save_steps=5000,
    evaluation_strategy='steps',
    eval_steps=5000,
    load_best_model_at_end=True,
    use_mps_device=torch.backends.mps.is_available(),
)

# Step 5: Instantiate a Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Step 6: Train the model
trainer.train(resume_from_checkpoint = True)
model.save_pretrained("d:/mymodels/gpt2_fine_tune")
tokenizer.save_pretrained("d:/mymodels/gpt2_fine_tune")
#trainer.save_model("d:/mymodels/gpt2_fine_tune")

