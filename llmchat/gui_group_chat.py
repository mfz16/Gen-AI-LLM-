# import re
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# model_path = "d:/mymodels/gpt2_fine_tune"
# mod = GPT2LMHeadModel.from_pretrained(model_path)
# token = GPT2Tokenizer.from_pretrained(model_path)
# token.add_special_tokens({'additional_special_tokens': ['[PAD]']})
# nlp = pipeline("text-generation", model=model_path)

# example = "Hello how are you"
# example1 = "Vijay Pune:kya kar rhe ho"

# generation_results = nlp(example1, max_length=200, num_return_sequences=1)
# print(generation_results)

# # Define a regex pattern to capture sender and message pairs
# pattern = r'(Dubey|Vijay Pune|RAVISH RANA|Faraz|Karan Gupta|Saurabh Dasgupta|Shashank Purohit),([^\\n]+)'

# matches = re.findall(pattern, str(generation_results))
# chat_history = []

# for m in matches:
#     sender, message = m
#     message = message.strip('"')
#     chat_history.append(f"{sender}: {message}")

# # Output the chat history
# for chat_entry in chat_history:
#     print(chat_entry)

from flask import Flask, request, render_template
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

app = Flask(__name)

# Load your GPT-2 model and tokenizer
model_path = "d:/mymodels/gpt2_fine_tune"
mod = GPT2LMHeadModel.from_pretrained(model_path)
token = GPT2Tokenizer.from_pretrained(model_path)
token.add_special_tokens({'additional_special_tokens': ['[PAD]']})
nlp = pipeline("text-generation", model=model_path)

@app.route("/", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_message = request.form["user_message"]

        # Generate a response
        generation_results = nlp(user_message, max_length=200, num_return_sequences=1)
        pattern = r'(Dubey|Vijay Pune|RAVISH RANA|Faraz|Karan Gupta|Saurabh Dasgupta|Shashank Purohit),([^\\n]+)'
        matches = re.findall(pattern, str(generation_results))
        chat_history = []

        for m in matches:
            sender, message = m
            message = message.strip('"')
            chat_history.append(f"{sender}: {message}")

        # Combine chat history into a single string
        chat_output = "\n".join(chat_history)

        return render_template("chat.html", user_message=user_message, chat_output=chat_output)

    return render_template("chat.html", user_message="", chat_output="")

if __name__ == "__main__":
    app.run(debug=True)
