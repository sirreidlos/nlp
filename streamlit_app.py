import streamlit as st
import torch
import gc
from transformers import MBartForConditionalGeneration
from tokenizer import IndoNLGTokenizer
from utils import DetoxificationEvaluator, generate_response

@st.cache_resource
def load_model():
    model = MBartForConditionalGeneration.from_pretrained('./bart-finetuned-toxicity2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model

@st.cache_resource
def load_tokenizer():
    return IndoNLGTokenizer.from_pretrained('./bart-finetuned-toxicity2')

@st.cache_resource
def load_evaluator():
    return DetoxificationEvaluator()

def safe_generate_response(model, tokenizer, prompt):
    try:
        # Clear GPU cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        response = generate_response(model, tokenizer, prompt)
        
        # Clear cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Error: Could not generate response"

def safe_evaluate(evaluator, prompt, response):
    try:
        return evaluator.evaluate_pair(prompt, response)
    except Exception as e:
        st.error(f"Error evaluating response: {str(e)}")
        return {"cosine_similarity": 0.0, "normalized_levenshtein": 0.0}

try:
    model = load_model()
    tokenizer = load_tokenizer()
    evaluator = load_evaluator()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Show title and description.
st.title("💬 Indonesian Text Detoxification")
st.write('''
This is a simple model that detoxify messages written in Indonesian It also
gives label to the output messages. `nontoxic` means the original text is safe,
`deleted` means part of the text is removed, `rewritten` means part of the text 
is rephrased, and `toxic` simply means the text's sentiment is too negative to 
be rewritten without removing context.

Do note that the label given is not always accurate.
'''
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Input text to detoxify"):
    prompt = f"DETOXIFY: {prompt}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Generating response..."):
        response = safe_generate_response(model, tokenizer, prompt)

    _, text_response = response.split(": ", 1)
    with st.spinner("Evaluating response..."):
        eval_result = safe_evaluate(evaluator, prompt, text_response)

    cosine_sim = eval_result["cosine_similarity"]
    levenshtein_sim = eval_result["normalized_levenshtein"]

    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown(f'<small style="color:#999;">Cosine Similarity: {cosine_sim:.2f}, Levenshtein Distance: {levenshtein_sim:.2f}</small>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()