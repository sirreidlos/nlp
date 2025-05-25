import streamlit as st

import torch
import shutil
import random
import numpy as np
import pandas as pd
from torch import optim
from transformers import MBartForConditionalGeneration
# from indobenchmark import IndoNLGTokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, Trainer, TrainingArguments
import Levenshtein

from transformers import BertTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
# from datasets import load_metric
import evaluate

from typing import Dict, List
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score


from tokenizer import IndoNLGTokenizer
from utils import DetoxificationEvaluator, generate_response

@st.cache_resource
def load_model():
    return MBartForConditionalGeneration.from_pretrained('./bart-finetuned-toxicity2')

@st.cache_resource
def load_tokenizer():
    return IndoNLGTokenizer.from_pretrained('./bart-finetuned-toxicity2')

def load_evaluator():
    return DetoxificationEvaluator()

model = load_model()
tokenizer = load_tokenizer()
evaluator = load_evaluator()

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("Input text to detoxify"):
    prompt = f"DETOXIFY: {prompt}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate a response using the OpenAI API.
    response = generate_response(model, tokenizer, prompt)
    eval_result = evaluator.evaluate_pair(prompt, response)

    cosine_sim = eval_result["cosine_similarity"]
    levenshtein_sim = eval_result["normalized_levenshtein"]

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown(f'<small style="color:#999;">Cosine Similarity: {cosine_sim}, Levenshtein Distance: {levenshtein_sim}</small>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
