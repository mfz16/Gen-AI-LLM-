import pandas as pd
import torch
<<<<<<< HEAD
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
torch.cuda.init()
print(torch.version.cuda)
=======
import torch_directml
dml = torch_directml.device()
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
#torch.backends.cudnn.enabled = False
#torch.cuda.is_available = lambda : False
#print(torch.version.cuda)
>>>>>>> a6cd15a4f81f2970c26d4883507f261a454eeabe
# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" or other variants if needed
tokenizer = GPT2Tokenizer.from_pretrained(model_name,device=dml)

# Add a special [NAME] token to the tokenizer
tokenizer.add_special_tokens({'additional_special_tokens': ['[NAME]']})

model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Load your custom group chat data from a CSV file
csv_file_path = "c:/test/chat/clean_chat.csv"
df = pd.read_csv(csv_file_path)

# Combine person's name and message, using [NAME] as a separator
chat_data = [f"[NAME] {name}: {message}" for name, message in zip(df["Sender"], df["Message"])]


# Tokenize the data
input_ids = tokenizer(chat_data, return_tensors="pt", padding=True, truncation=True)

# Create a custom dataset with labels for next-token prediction
class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.input_ids[idx]}

chat_dataset = ChatDataset(input_ids)
# Tokenize and format the data within the dataset
def tokenize_function(examples):
    return tokenizer(examples, return_tensors="pt", padding=True, truncation=True)



data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="c:/mymodels/fine_tuned_model/group-chat-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,  # You can adjust this
    per_device_train_batch_size=1,  # You can adjust this
    save_steps=10_000,  # You can adjust this
    evaluation_strategy="steps",
    save_total_limit=2,
    learning_rate=2e-5,  # You can adjust this
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=chat_dataset,
)
# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("c:/mymodels/fine-tuned-group-chatbot")
tokenizer.save_pretrained("c:/mymodels/fine-tuned-group-chatbot")