# simulation/simulate_chat_tests.py
# =================================
## python -m simulation.simulate_chat_tests
import argparse
import time
import pandas as pd
import uuid
import csv
import sys
import types
from datetime import datetime, timezone
import torch
import logging
from pathlib import Path
from services.mongo import db
from services.monitoring import log_execution

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

test_results_col = db["test_results"]
monitoring_col = db["monitoring"]

faq = db["faq"]
default_chat = db["default_chat"]
knowledge = db["knowledge"]


def log_event(event_type, details, status="success", log_source="simulation"):
    monitoring_col.insert_one({
        "event": event_type,
        "details": details,
        "status": status,
        "log_source": log_source,
        "timestamp": datetime.now(timezone.utc)
    })


def load_test_queries(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found!")
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_bert_best_match(query, kb_entries):
    from pages.Chatbot import get_bert_embeddings
    query_emb = get_bert_embeddings(query)
    best_score = -1
    best_answer = "No match found."
    for entry in kb_entries:
        entry_text = f"{entry.get('question', '')} {entry.get('answer', '')}"
        entry_emb = get_bert_embeddings(entry_text)
        score = torch.cosine_similarity(
            torch.tensor(query_emb), 
            torch.tensor(entry_emb), 
            dim=0
        ).item()
        if score > best_score:
            best_score = score
            best_answer = entry.get("answer", "No answer available.")
    return best_answer, best_score


def simulate_test(model_name, questions, execution_type="test"):
    start_time = time.time()
    
    time.sleep(0.1 * len(questions))
    score = round(0.7 + (0.3 * len(questions) / 100), 4) 

    execution_time = round(time.time() - start_time, 4)
    return {"model": model_name, "execution_time": execution_time, "score": score}

class SessionStateMock:
    def __init__(self):
        self._store = {}
    def __getattr__(self, item):
        try:
            return self._store[item]
        except KeyError:
            raise AttributeError(f"No attribute {item}")
    def __setattr__(self, key, value):
        if key == "_store":
            super().__setattr__(key, value)
        else:
            self._store[key] = value
    def __getitem__(self, key):
        return self._store.get(key, None)
    def __setitem__(self, key, value):
        self._store[key] = value
    def get(self, key, default=None):
        return self._store.get(key, default)
    def __contains__(self, key):
        return key in self._store


def no_op_decorator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

class DummyContextManager:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

fake_st = types.SimpleNamespace(
    subheader=lambda *a, **k: None,
    progress=lambda *a, **k: None,
    session_state=SessionStateMock(),
    cache_data=no_op_decorator,
    cache_resource=no_op_decorator,
    set_page_config=lambda *a, **k: None,
    form=lambda *a, **k: DummyContextManager(),
    text_input=lambda *a, **k: "",
    text_area=lambda *a, **k: "",
    form_submit_button=lambda *a, **k: False,
    error=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    success=lambda *a, **k: None,
    radio=lambda *a, **k: None,
    button=lambda *a, **k: False,
    chat_input=lambda *a, **k: None,
    markdown=lambda *a, **k: None,
    sidebar=types.SimpleNamespace(
        button=lambda *a, **k: False,
        markdown=lambda *a, **k: None,
        expander=lambda *a, **k: fake_st,
        caption=lambda *a, **k: None
    )
)
sys.modules['streamlit'] = fake_st


def run_tests(questions_file="data/questions_short.txt"):
    from pages.Chatbot import generate_gpt3_reply, get_bert_embeddings

    TEST_QUERIES = load_test_queries(questions_file)
    print(f"File with {len(TEST_QUERIES)} queries loaded.")

    logging.info(f"Starting automated chatbot tests with {len(TEST_QUERIES)} queries...")
    user_id = f"test_{uuid.uuid4().hex[:8]}"

    kb_entries = []
    for doc in db["faq"].find({}, {"_id": 0, "question": 1, "answer": 1}):
        kb_entries.append({
            "question": doc.get("question", ""), 
            "answer": doc.get("answer", "")})
    intent_doc = db["default_chat"].find_one({
        "intents": {"$exists": True}}, 
        {"_id": 0, "intents": 1})
    if intent_doc:
        for intent in intent_doc.get("intents", []):
            for pattern in intent.get("patterns", []):
                for response in intent.get("responses", []):
                    kb_entries.append({
                        "question": pattern, 
                        "answer": response})
    for doc in db["knowledge"].find({}, {"_id": 0, "title": 1, "content": 1}):
        kb_entries.append({
            "question": doc.get("title", ""), 
            "answer": doc.get("content", "")})

    results = []

    total_queries = len(TEST_QUERIES)
    for i, query in enumerate(TEST_QUERIES, start=1):
        print(f"Processing {i}/{len(TEST_QUERIES)}: {query}")

        progress = int((i / total_queries) * 100)
        logging.info(f"Progress: {progress}% ({i}/{total_queries})")

        try:
            import streamlit as st
            st.progress(progress)
        except ImportError:
            pass

        # --- 🔹 GPT-3 Test ---
        start_time = time.time()
        gpt3_response = generate_gpt3_reply(query)
        gpt3_time = round(time.time() - start_time, 3)

        # --- 🔹 BERT Test ---
        start_time = time.time()
        bert_response, bert_score = get_bert_best_match(query, kb_entries)
        bert_time = round(time.time() - start_time, 3)

        try:
            gpt3_emb = get_bert_embeddings(gpt3_response)
            bert_ref_emb = get_bert_embeddings(bert_response)
            gpt3_score = torch.cosine_similarity(
                torch.tensor(gpt3_emb),
                torch.tensor(bert_ref_emb),
                dim=0
            ).item()
        except Exception:
            gpt3_score = None

        test_results_col.insert_one({
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "query": query,
            "response": gpt3_response,
            "score": gpt3_score,
            "execution_time": gpt3_time,
            "model": "gpt-3",
            "log_source": "simulation"
        })

        test_results_col.insert_one({
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "query": query,
            "response": bert_response,
            "score": bert_score,
            "execution_time": bert_time,
            "model": "bert",
            "log_source": "simulation"
        })

        results.append({
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "query": query,
            "gpt3_response": gpt3_response,
            "gpt3_score": gpt3_score,
            "bert_response": bert_response,
            "bert_score": bert_score,
            "gpt3_time": gpt3_time,
            "bert_time": bert_time
        })
        
        log_event("test_case", {"query": query}, log_source="simulation")
                
        logging.info(f"Query: {query}")
        logging.info(f"GPT-3 [{gpt3_time}s, score={gpt3_score}]: {gpt3_response}")
        logging.info(f"BERT [{bert_time}s, score={bert_score:.3f}]: {bert_response}")
        logging.info("-" * 50)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    csv_filename = f"simulation/result/test_results-{timestamp}.csv"
    Path("simulation/result").mkdir(parents=True, exist_ok=True)
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "user_id", "timestamp", "query",
            "gpt3_response", "gpt3_score", "gpt3_time",
            "bert_response", "bert_score", "bert_time"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    logging.info(f"✅ Test results saved to {csv_filename}")
    logging.info(f"✅ Test logs saved to MongoDB collections 'test_results', and 'monitoring'")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", 
        default="data/questions_short.txt", 
        help="Path to questions file"
    )
    args = parser.parse_args()
    run_tests(args.file)

    test_questions = [
        "How do I reset my password?", 
        "Where is my invoice?", 
        "Can I change my subscription?"
    ]

    results = []
    results.append(simulate_test("BERT", test_questions, "test"))
    results.append(simulate_test("GPT3", test_questions, "test"))

    df_results = pd.DataFrame(results)
    print(df_results)