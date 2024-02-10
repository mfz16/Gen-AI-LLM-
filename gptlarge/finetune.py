import torch
from torch.utils.data import random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, \
    DataCollatorForLanguageModeling

# Step 1: Load the pre-trained GPT-2 model
model_checkpoint='d:\huggingface_models\gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

# Step 2: Tokenize the training data
tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({'additional_special_tokens': ['[NAME]']})
tokenizer.add_special_tokens({'additional_special_tokens': ['[MESSAGE]']})
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
with open("d:/test/chat/chat_data_tok.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Print the first few lines of the dataset
for line in lines[:5]:
    print(line)

# Tokenize and print the first few samples
for line in lines[:5]:
    tokens = tokenizer(line, return_tensors="pt")
    print(tokens)

# Step 3: Prepare the training data
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="d:/test/chat/chat_in_txt.txt",
    block_size=10
)
train_size = int(0.9 * len(dataset))
eval_size = len(dataset) - train_size

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
print("Training dataset size:", len(train_dataset))
print("Evaluation dataset size:", len(eval_dataset))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")

# Step 4: Create a TrainingArguments object
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
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
trainer.train()
#trainer.train(resume_from_checkpoint = True)
model.save_pretrained("d:/mymodels/gpt2_large_fine_tune")
tokenizer.save_pretrained("d:/mymodels/gpt2_large_fine_tune")
#trainer.save_model("d:/mymodels/gpt2_fine_tune")
