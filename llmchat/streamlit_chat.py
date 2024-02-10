import streamlit as st
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch

# Load your GPT-2 model and tokenizer
model_path = "d:/mymodels/gpt2_fine_tune"
mod = GPT2LMHeadModel.from_pretrained(model_path)
token = GPT2Tokenizer.from_pretrained(model_path)
token.add_special_tokens({'additional_special_tokens': ['[PAD]']})
nlp = pipeline("text-generation", model=model_path)

# Initialize chat history
chat_history = []

st.set_page_config(
    page_title="WhatsApp Group Chat",
    page_icon=":iphone:",
    layout="wide"
)

st.title("WhatsApp Group Chat")

# Define a user input field and sender name input
user_message = st.text_input("Type your message:")
sender_name = st.text_input("Your Name:")

def get_response(message):
    generation_results = nlp(message, max_length=200, num_return_sequences=1)
    import re

    words = re.split('Dubey,|Vijay Pune,|RAVISH RANA,|Faraz,|Karan Gupta,', str(generation_results))
    pattern= r'(Dubey|Vijay Pune|RAVISH RANA|Faraz|Karan Gupta|Saurabh Dasgupta|Shashank Purohit),([^\\n]+)'

    matches = re.findall(pattern, str(generation_results))
    print(f"matches is {matches}")
    msg_list = []
    for m in matches:
        sender, message = m
        message = message.strip('"')
        msg_list.append(f"{sender}: {message}")
        print(f"{sender}: {message}")
    return msg_list

if st.button("Send"):
    if user_message and sender_name:
        message = f"{sender_name}: {user_message}"
        chat_history.append(message)
        chat_history.extend(get_response(message))
        #print(get_response(message))
print(chat_history)

# Display the chat interface
st.subheader("Group Chat")

# Render the chat history as a list
for chat_message in chat_history:
    st.text(chat_message)

# Add some CSS to style the chat interface
st.markdown(
    """
    <style>
    .st-ax {
        background-color: #F4F4F4;
        padding: 10px;
        border-radius: 10px;
    }
    .st-ay {
        background-color: #25d366;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
