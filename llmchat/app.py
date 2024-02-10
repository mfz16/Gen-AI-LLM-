from flask import Flask, request, render_template
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
print(torch.cuda.is_available())
app = Flask(__name__)

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
        sender_name= request.form["sender_name"]
        message = f"{sender_name}: {user_message}"

        # Generate a response
        generation_results = nlp(message, max_length=100, num_return_sequences=1)
        print(f"my message: {message}")
        print(f"response: {generation_results}")
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