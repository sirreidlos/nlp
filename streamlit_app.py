import streamlit as st
from transformers import MBartForConditionalGeneration
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
    
    response = generate_response(model, tokenizer, prompt)
    eval_result = evaluator.evaluate_pair(prompt, response)

    cosine_sim = eval_result["cosine_similarity"]
    levenshtein_sim = eval_result["normalized_levenshtein"]

    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown(f'<small style="color:#999;">Cosine Similarity: {cosine_sim:.2f}, Levenshtein Distance: {levenshtein_sim:.2f}</small>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
