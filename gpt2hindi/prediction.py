from transformers import GPT2LMHeadModel, GPT2Tokenizer,TrainingArguments, Trainer, TextDataset,DataCollatorForLanguageModeling
import torch
import transformers
from transformers import pipeline
model_path = "d:\mymodels\hindi\gpt2_fine_tune"
mod=GPT2LMHeadModel.from_pretrained(model_path)
token=GPT2Tokenizer.from_pretrained(model_path)
token.add_special_tokens({'additional_special_tokens': ['[PAD]']})
nlp = pipeline("text-generation",model=model_path)
example = "Hello how are you"
example1="vijay"
generation_results = nlp(example1, max_length=200, num_return_sequences=1)
print(generation_results)