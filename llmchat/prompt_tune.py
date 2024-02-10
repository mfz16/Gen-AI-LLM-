from transformers import BertForNextSentencePrediction,BertTokenizer
#model = BertForNextSentencePrediction.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', cache_dir="D:/pretrain_models/next_sentence/huggingface_cache/")
#tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', cache_dir="D:/pretrain_models/next_sentence/huggingface_cache/")
from transformers import GPT2LMHeadModel, GPT2Tokenizer,TrainingArguments, Trainer, TextDataset,DataCollatorForLanguageModeling
cache_dir = "D:/pretrain_models/next_sentence/huggingface_cache/"
#model = GPT2LMHeadModel.from_pretrained('gpt2',cache_dir=cache_dir)
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2',cache_dir=cache_dir)

import torch
import transformers
from transformers import pipeline


model_path = "d:/pretrain_models/next_sentence/huggingface_cache/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10"
mod=GPT2LMHeadModel.from_pretrained(model_path)
token=GPT2Tokenizer.from_pretrained(model_path)
token.add_special_tokens({'additional_special_tokens': ['[PAD]']})

nlp = pipeline("text-generation",model=model_path)
example = "Hello how are you"
generation_results = nlp(example, max_length=50, num_return_sequences=1)
print(generation_results)
# Fine-tuning configuration
my_data="d:/test/chat/clean_chat.csv"
import pandas as pd
df=pd.read_csv(my_data)
names = df["Sender"]
messages = df["Message"]

# Combine user messages and model responses
chat_data = [f"[NAME] {name}: {message}" for name, message in zip(df["Sender"], df["Message"])]

# Tokenize and format the data within the dataset
def tokenize_function(examples):
    return token(examples, return_tensors="pt", padding=False, truncation=True)


# input_data = []
# for i in range(0, len(chat_data), 2):
#     user_message = chat_data[i].split("User: ")[1]
#     model_response = chat_data[i + 1].split("Model: ")[1]
#     input_data.append(f"User: {user_message}\nModel: {model_response}")


input_ids = token(my_data, return_tensors="pt", padding=True, truncation=True)
class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, tokenizer):
        self.input_ids = input_ids["input_ids"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_text = self.tokenizer.decode(self.input_ids[idx])
        return {"input_ids": self.input_ids[idx], "labels": self.tokenizer.encode(input_text, add_special_tokens=False)}

chat_dataset = ChatDataset(input_ids, token)
dataset = TextDataset(
    tokenizer=token,
    file_path=my_data,
    block_size=128,  # Adjust the block size as needed
    overwrite_cache=False,
    
)
data_collator = DataCollatorForLanguageModeling(tokenizer=token, mlm=False)
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    output_dir="d:/mymodels/fine_tuned_model",
    overwrite_output_dir=True,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=2e-5
)

# Create a Trainer for fine-tuning
trainer = Trainer(
    model=mod,
    args=training_args,
    data_collator=None,
    train_dataset=dataset
)

# Fine-tune the model
trainer.train()

mod.save_pretrained("d:/mymodels/fine-tuned-group-chatbot")
token.save_pretrained("d:/mymodels/fine-tuned-group-chatbot")





# from langchain.llms import HuggingFacePipeline

# llm = HuggingFacePipeline.from_model_id(
#     model_id="gpt2",
#     task="text-generation",
#     model_kwargs={"temperature": .6, "max_length": 64},
# )
# from langchain.prompts import PromptTemplate

# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)

# chain = prompt | llm

# question = "What is electroencephalography?"

# print(chain.invoke({"question": question}))