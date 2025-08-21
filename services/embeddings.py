# services/embeddings.py
# ===============================
## python -m services.embeddings

import os
import logging
from functools import lru_cache
from typing import Union, List
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from openai import OpenAI, RateLimitError, APIError

from dotenv import load_dotenv
load_dotenv(dotenv_path="config/.env")

# ===============================
# OpenAI API Client
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "❌ OPENAI_API_KEY is not defined.\n"
        "➡ Check if you exported the variable in the shell with:\n"
        "   export OPENAI_API_KEY='uour-key'\n"
        "➡ Or set it in the config/.env file like this:\n"
        "   OPENAI_API_KEY=sk-xxxx..."
    )
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)

# =========================
# Device configuration
# =========================
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"🔹 Using device: {DEFAULT_DEVICE}")

# =========================
# Auxiliary function: truncation warning
# =========================
def check_truncation(tokenizer, texts: List[str], max_length: int):
    for t in texts:
        if len(tokenizer.encode(t, truncation=False)) > max_length:
            logging.warning(f"⚠ Truncated text for {max_length} tokens: {t[:50]}...")


# =========================
# HuggingFace BERT
# =========================
@lru_cache(maxsize=1)
def load_bert(device=DEFAULT_DEVICE):
    logging.info("🔹 Loading model BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


def get_bert_embeddings(texts: Union[str, List[str]], device=DEFAULT_DEVICE) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts]
    tokenizer, model = load_bert()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# =========================
# OpenAI GPT Embeddings
# =========================
def get_gpt3_embeddings(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]

    try:
        res = client.embeddings.create(model=model, input=texts)
        return np.array([d.embedding for d in res.data])
    except RateLimitError as e:
        logging.error("❌ OpenAI quota exceeded or billing missing.")
        return np.zeros((len(texts), 1536))


# =========================
# Sentence-BERT (SBERT)
# =========================
@lru_cache(maxsize=1)
def load_sbert():
    logging.info("🔹 Loading model SBERT...")
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_sbert_embeddings(texts: Union[str, List[str]], device=DEFAULT_DEVICE) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    model = load_sbert()
    return np.array(model.encode(texts, convert_to_numpy=True))


# =========================
# Embedding benchmark utility
# =========================
def benchmark_embedding(model_name: str, embed_fn, texts: List[str], **kwargs):
    import time
    start = time.time()
    emb = embed_fn(texts, **kwargs) 
    elapsed = time.time() - start
    logging.info(f"⏱ {model_name} - Time: {elapsed:.3f}s - Shape: {emb.shape}")
    return emb, elapsed

def benchmark_all_models(texts: List[str], device=DEFAULT_DEVICE):
    logging.info("📊 Running benchmark of all models...")
    results = {}

    model_fns = {
        "BERT": lambda txts: get_bert_embeddings(txts, device=device),
        "GPT-3": lambda txts: get_gpt3_embeddings(txts),
        "SBERT": lambda txts: get_sbert_embeddings(txts, device=device),
    }

    for name, fn in model_fns.items():
        _, elapsed = benchmark_embedding(name, fn, texts)
        results[name] = elapsed

    plt.bar(results.keys(), results.values(), color=["blue", "green", "orange"])
    plt.ylabel("Time (seconds)")
    plt.title("Embeddings Benchmark")
    plt.show()
    return results


# =========================
# Local execution for testing
# =========================
if __name__ == "__main__":
    logging.info("Running local tests on embeddings.py...")
    sample_text = ["How can I reset my password?"]
    benchmark_all_models(sample_text)