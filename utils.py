import torch
import Levenshtein
from transformers import BertTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity



class DetoxificationEvaluator:
    def __init__(self, model_name="indobenchmark/indobert-base-p1"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.prefixes = [
            "DETOXIFY: ",
            "NONTOXIC: ",
            "DELETED: ",
            "REWRITTEN: ",
            "TOXIC: "
        ]

    def strip_prefix(self, text):
        for prefix in self.prefixes:
            if text.startswith(prefix):
                return text[len(prefix):].strip()
        return text.strip()

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to("cpu")
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze()


    def get_cosine_sim(self, text1, text2):
        emb1 = self.get_embedding(text1).unsqueeze(0).numpy()
        emb2 = self.get_embedding(text2).unsqueeze(0).numpy()
        return cosine_similarity(emb1, emb2)[0][0]

    def get_normalized_levenshtein(self, text1, text2):
        dist = Levenshtein.distance(text1, text2)
        max_len = max(len(text1), len(text2))
        return 1.0 - dist / max_len if max_len > 0 else 1.0

    def evaluate_pair(self, input_text, output_text):
        stripped_input = self.strip_prefix(input_text)
        stripped_output = self.strip_prefix(output_text)
        cosine_sim = self.get_cosine_sim(stripped_input, stripped_output)
        lev_sim = self.get_normalized_levenshtein(stripped_input, stripped_output)
        return {
            "cosine_similarity": cosine_sim,
            "normalized_levenshtein": lev_sim
        }

    def evaluate_batch(self, pairs):
        results = []
        for input_text, output_text in pairs:
            scores = self.evaluate_pair(input_text, output_text)
            results.append(scores)
        return results


def generate_response(model, tokenizer, input_text):
    inputs = tokenizer.prepare_input_for_generation(
        inputs=input_text,
        model_type='indobart',
        lang_token='[indonesian]',
        return_tensors='pt',
        padding='longest',
    )

    device = next(model.parameters()).device

    # IndoNLGTokenizer does not automatically add batch dim if input is a string
    input_ids = inputs["input_ids"].unsqueeze(0) if inputs["input_ids"].dim() == 1 else inputs["input_ids"]
    attention_mask = inputs["attention_mask"].unsqueeze(0) if inputs["attention_mask"].dim() == 1 else inputs["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

transformation_prefix = {
    0: "NONTOXIC: ",
    1: "DELETED: ",
    2: "REWRITTEN: ",
    3: "TOXIC: "
}

# Reverse map prefix string to class label for evaluation
prefix_to_class = {v: k for k, v in transformation_prefix.items()}

def extract_prefix(text):
    for prefix in prefix_to_class:
        if text.startswith(prefix):
            return prefix
    return None
