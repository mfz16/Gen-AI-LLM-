from transformers import GPT2LMHeadModel, GPT2Tokenizer,TrainingArguments, Trainer, TextDataset,DataCollatorForLanguageModeling
from transformers import pipeline
model_path = "d:/mymodels/gpt2_fine_tune"
mod=GPT2LMHeadModel.from_pretrained(model_path)
token=GPT2Tokenizer.from_pretrained(model_path)
token.add_special_tokens({'additional_special_tokens': ['[PAD]']})
nlp = pipeline("text-generation",model=model_path)
example = "Hello how are you"
example1="Vijay Pune:kya kar rhe ho"
generation_results = nlp(example1, max_length=200, num_return_sequences=1)
print(generation_results)
import re

words = re.split('Dubey,|Vijay Pune,|RAVISH RANA,|Faraz,|Karan Gupta,', str(generation_results))
for text in words:
 #print(re.findall("*", text))
 match = re.search(r'"(.*?)"', text)
 print(match)
#out=str(generation_results).split("Dubey:","Vijay Pune:")
print(f"out is {words}")
#pattern = r'(Dubey|Vijay Pune|RAVISH RANA|Faraz|Karan Gupta)(.*?)(?=\d+,|\Z)'
#pattern = r'(Dubey|Vijay Pune|RAVISH RANA|Faraz|Karan Gupta),"(.*?)"(?=\d+,|\Z)' #r'(\d+),([A-Za-z\s]+):"(.*?)"'
#pattern = r'([^,]+),"([^\\n]+)"'
pattern= r'(Dubey|Vijay Pune|RAVISH RANA|Faraz|Karan Gupta|Saurabh Dasgupta|Shashank Purohit),([^\\n]+)'

matches = re.findall(pattern, str(generation_results))
print(f"matches is {matches}")
for m in matches:
    sender, message = m
    message = message.strip('"')
    print(f"{sender}: {message}")
#     print()