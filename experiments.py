# experiments.py
# Unified pipeline to generate baseline (TF-IDF), BERT/SBERT (HF) and GPT-3 embeddings
# for classification (priority/sentiment) and clustering, saving ready-to-use artefacts.
# Author: Brenda's ARP toolkit
# Python >= 3.9 recommended

import os
import sys
import json
import time
import math
import argparse
import pathlib
import logging
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

# Optional imports (loaded on-demand)
# transformers / sentence_transformers / openai are imported when needed.

# ----------------------------
# Paths & logging
# ----------------------------
ROOT = pathlib.Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

# ----------------------------
# Utilities
# ----------------------------
def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_json(obj, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def plot_cm(cm: np.ndarray, labels: List[str], title: str, out_path: pathlib.Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

def plot_embed_2d(X: np.ndarray, y: Optional[List]=None, method: str="tsne", out_path: pathlib.Path=None, title: str="Embedding"):
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, init="random", perplexity=30)
    Z = reducer.fit_transform(X)
    plt.figure(figsize=(6, 5))
    if y is None:
        plt.scatter(Z[:,0], Z[:,1], s=8)
    else:
        # simple colour mapping by class index
        _, inv = np.unique(y, return_inverse=True)
        plt.scatter(Z[:,0], Z[:,1], s=8, c=inv, cmap="tab10")
    plt.title(f"{title} ({method.upper()})")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=140)
        plt.close()
    else:
        plt.show()

def ensure_series(obj) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj
    return pd.Series(obj)

# ----------------------------
# Data loaders
# ----------------------------
def load_local_csv(csv_path: str, text_col: str, label_col: Optional[str]=None) -> Tuple[pd.Series, Optional[pd.Series]]:
    df = pd.read_csv(csv_path)
    assert text_col in df.columns, f"Text column '{text_col}' not found in CSV."
    X = df[text_col].astype(str).fillna("")
    y = None
    if label_col:
        assert label_col in df.columns, f"Label column '{label_col}' not found in CSV."
        y = df[label_col].astype(str)
        y = y.replace({
            "Critical": "High",   # merge Critical into High
            "critical": "High",   # in case it comes lowercase
        })
    return X, y

def load_hf_dataset(hf_name: str, text_col: str, label_col: Optional[str]=None, split: str="train") -> Tuple[pd.Series, Optional[pd.Series]]:
    from datasets import load_dataset
    ds = load_dataset(hf_name, split=split)
    X = pd.Series([str(x) for x in ds[text_col]])
    y = None
    if label_col:
        raw = ds[label_col]
        # map class indices to names if feature has names
        try:
            names = ds.features[label_col].names  # may exist for ClassLabel
            y = pd.Series([names[i] if isinstance(i, int) and 0 <= i < len(names) else str(i) for i in raw])
        except Exception:
            y = pd.Series([str(i) for i in raw])
    return X, y

# ----------------------------
# Embedders
# ----------------------------
def embed_tfidf(train_texts: List[str], test_texts: List[str], max_features: int=20000, min_df: int=2):
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec

def embed_sbert(texts: List[str], model_name: str="sentence-transformers/all-MiniLM-L6-v2", batch_size: int=256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    X = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalise_embeddings=True)
    return X

def embed_bert_meanpool(texts: List[str], model_name: str="bert-base-uncased", batch_size: int=16, max_length: int=256) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    all_vecs = []
    mdl.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = mdl(**enc).last_hidden_state  # [B, T, H]
            # mean pooling (mask-aware)
            mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            summed = (out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            emb = (summed / counts).cpu().numpy()
            # L2 normalise
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms
            all_vecs.append(emb)
    return np.vstack(all_vecs)

def embed_openai(texts: List[str], model: str="text-embedding-3-small", batch_size: int=2048) -> np.ndarray:
    # Requires OPENAI_API_KEY in environment or .env
    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI
    client = OpenAI()
    vecs: List[List[float]] = []
    # OpenAI API accepts large batches; we will slice to be safe
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        vecs.extend([d.embedding for d in resp.data])
    X = np.array(vecs, dtype=np.float32)
    # normalise
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def build_embeddings(texts: List[str], embedder: str) -> np.ndarray:
    embedder = embedder.lower()
    if embedder == "sbert":
        return embed_sbert(texts)
    if embedder == "bert":
        return embed_bert_meanpool(texts)
    if embedder in ("gpt3", "openai"):
        return embed_openai(texts)
    raise ValueError(f"Unknown embedder: {embedder}")

# ----------------------------
# Tasks
# ----------------------------
def task_baseline_tfidf(
    X_text: pd.Series,
    y: pd.Series,
    test_size: float,
    out_prefix: str
) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=42, stratify=y
    )
    pipe = make_pipeline(
        TfidfVectorizer(max_features=20000, min_df=2),
        SGDClassifier(loss="log_loss", max_iter=1000, class_weight="balanced", random_state=42)
    )
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_sec = time.time() - t0

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    report_csv = REPORTS_DIR / f"{out_prefix}_baseline_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_csv)
    labels = sorted(map(str, y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_png = REPORTS_DIR / f"{out_prefix}_baseline_cm.png"
    plot_cm(cm, labels, "Confusion Matrix - Baseline (TF-IDF + SGD)", cm_png)

    preds_csv = REPORTS_DIR / f"{out_prefix}_baseline_predictions_sample.csv"
    pd.DataFrame({
        "text": X_test.iloc[:200].tolist(),
        "true_label": y_test.iloc[:200].tolist(),
        "pred_label": y_pred[:200]
    }).to_csv(preds_csv, index=False)

    # Save pipeline
    model_path = MODELS_DIR / f"{out_prefix}_baseline_tfidf_sgd.joblib"
    joblib.dump(pipe, model_path)

    meta = {
        "accuracy": acc,
        "macro_f1": f1m,
        "train_seconds": train_sec,
        "report_csv": str(report_csv),
        "confusion_matrix_png": str(cm_png),
        "predictions_csv": str(preds_csv),
        "model_path": str(model_path),
    }
    save_json(meta, REPORTS_DIR / f"{out_prefix}_baseline_summary.json")
    logging.info(f"[Baseline] acc={acc:.4f} macroF1={f1m:.4f} | saved under prefix '{out_prefix}'")

def task_classify_with_embeddings(
    X_text: pd.Series,
    y: pd.Series,
    embedder: str,
    test_size: float,
    out_prefix: str
) -> None:
    # Split first (to avoid leakage if embedder is train-time)
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=42, stratify=y
    )

    # Build embeddings
    t0 = time.time()
    if embedder.lower() == "tfidf":
        X_train, X_test, vec = embed_tfidf(X_train_txt.tolist(), X_test_txt.tolist())
        embed_seconds = time.time() - t0
        emb_dim = X_train.shape[1]
        embedder_used = "TF-IDF"
    else:
        X_train = build_embeddings(X_train_txt.tolist(), embedder)
        X_test = build_embeddings(X_test_txt.tolist(), embedder)
        embed_seconds = time.time() - t0
        emb_dim = X_train.shape[1]
        embedder_used = embedder.upper()

    # Classifier on embeddings
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)
    t1 = time.time()
    clf.fit(X_train, y_train)
    train_seconds = time.time() - t1

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    # Save artefacts
    report_csv = REPORTS_DIR / f"{out_prefix}_{embedder}_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_csv)

    labels = sorted(map(str, y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_png = REPORTS_DIR / f"{out_prefix}_{embedder}_cm.png"
    plot_cm(cm, labels, f"Confusion Matrix - {embedder_used}", cm_png)

    preds_csv = REPORTS_DIR / f"{out_prefix}_{embedder}_predictions_sample.csv"
    pd.DataFrame({
        "text": X_test_txt.iloc[:200].tolist(),
        "true_label": y_test.iloc[:200].tolist(),
        "pred_label": y_pred[:200]
    }).to_csv(preds_csv, index=False)

    # 2D plots of embeddings (test set)
    try:
        tsne_png = REPORTS_DIR / f"{out_prefix}_{embedder}_TSNE.png"
        plot_embed_2d(ensure_series(X_test).to_numpy() if hasattr(X_test, "toarray") else X_test,
                      y_test.tolist(), "tsne", tsne_png, f"{embedder_used} embeddings")
    except Exception as e:
        logging.warning(f"t-SNE plot failed: {e}")

    try:
        pca_png = REPORTS_DIR / f"{out_prefix}_{embedder}_PCA.png"
        plot_embed_2d(ensure_series(X_test).to_numpy() if hasattr(X_test, "toarray") else X_test,
                      y_test.tolist(), "pca", pca_png, f"{embedder_used} embeddings")
    except Exception as e:
        logging.warning(f"PCA plot failed: {e}")

    # Save classifier (and vectoriser if TF-IDF)
    model_path = MODELS_DIR / f"{out_prefix}_{embedder}_clf.joblib"
    joblib.dump(clf, model_path)
    vec_path = None
    if embedder.lower() == "tfidf":
        vec_path = MODELS_DIR / f"{out_prefix}_{embedder}_vectoriser.joblib"
        joblib.dump(vec, vec_path)

    meta = {
        "embedder": embedder_used,
        "embedding_seconds": embed_seconds,
        "embedding_dim": int(emb_dim),
        "train_seconds": train_seconds,
        "accuracy": acc,
        "macro_f1": f1m,
        "report_csv": str(report_csv),
        "confusion_matrix_png": str(cm_png),
        "predictions_csv": str(preds_csv),
        "tsne_png": str(tsne_png) if 'tsne_png' in locals() else None,
        "pca_png": str(pca_png) if 'pca_png' in locals() else None,
        "model_path": str(model_path),
        "vectoriser_path": str(vec_path) if vec_path else None
    }
    save_json(meta, REPORTS_DIR / f"{out_prefix}_{embedder}_summary.json")
    logging.info(f"[{embedder_used}] acc={acc:.4f} macroF1={f1m:.4f} | saved under prefix '{out_prefix}'")

def task_cluster_with_embeddings(
    X_text: pd.Series,
    embedder: str,
    k: int,
    out_prefix: str
) -> None:
    X = build_embeddings(X_text.tolist(), embedder)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)

    # Save clustering labels
    clusters_csv = REPORTS_DIR / f"{out_prefix}_{embedder}_clusters.csv"
    pd.DataFrame({"text": X_text, "cluster": labels}).to_csv(clusters_csv, index=False)

    # 2D plots
    try:
        tsne_png = REPORTS_DIR / f"{out_prefix}_{embedder}_clusters_TSNE.png"
        plot_embed_2d(X, labels.tolist(), "tsne", tsne_png, f"{embedder.upper()} clusters")
    except Exception as e:
        logging.warning(f"t-SNE plot failed: {e}")

    try:
        pca_png = REPORTS_DIR / f"{out_prefix}_{embedder}_clusters_PCA.png"
        plot_embed_2d(X, labels.tolist(), "pca", pca_png, f"{embedder.upper()} clusters")
    except Exception as e:
        logging.warning(f"PCA plot failed: {e}")

    meta = {
        "k": int(k),
        "silhouette": float(sil),
        "davies_bouldin": float(dbi),
        "clusters_csv": str(clusters_csv),
        "tsne_png": str(tsne_png) if 'tsne_png' in locals() else None,
        "pca_png": str(pca_png) if 'pca_png' in locals() else None
    }
    save_json(meta, REPORTS_DIR / f"{out_prefix}_{embedder}_clusters_summary.json")
    logging.info(f"[{embedder.upper()} clustering] k={k} | silhouette={sil:.4f} | DBI={dbi:.4f}")

# ----------------------------
# CLI
# ----------------------------
def main():
    seed_all(42)
    ap = argparse.ArgumentParser(description="Unified experiments for Baseline, BERT/SBERT, and GPT-3 embeddings.")
    ap.add_argument("--task", choices=["baseline", "classify", "cluster"], required=True,
                    help="baseline: TF-IDF+SGD; classify: embeddings + logistic regression; cluster: embeddings + KMeans")
    ap.add_argument("--dataset", choices=["local", "hf"], required=True)

    # Local CSV
    ap.add_argument("--csv-path", type=str, help="Path to local CSV when dataset=local")
    ap.add_argument("--text-col", type=str, help="Text column name (local CSV)")
    ap.add_argument("--label-col", type=str, default=None, help="Label column (optional for cluster)")

    # HF dataset
    ap.add_argument("--hf-name", type=str, help="HF dataset name (e.g., 'banking77')")
    ap.add_argument("--hf-text-col", type=str, help="Text column in HF dataset")
    ap.add_argument("--hf-label-col", type=str, default=None, help="Label column in HF dataset (optional for cluster)")
    ap.add_argument("--hf-split", type=str, default="train")

    # Embeddings / modelling
    ap.add_argument("--embedder", type=str, default="tfidf",
                    help="tfidf | bert | sbert | gpt3 (used for classify/cluster)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=3, help="K for KMeans (cluster task)")

    # Output prefix
    ap.add_argument("--out-prefix", type=str, default="exp",
                    help="Prefix used for artefact filenames")

    args = ap.parse_args()

    # Load data
    if args.dataset == "local":
        assert args.csv_path is not None, "Please provide --csv-path for local CSV."
        csv_path = args.csv_path
        assert args.text_col, "Please provide --text-col for local CSV."
        X, y = load_local_csv(csv_path, args.text_col, args.label_col)

    else:
        assert args.hf_name and args.hf_text_col, "Please provide --hf-name and --hf-text-col for HF."
        X, y = load_hf_dataset(args.hf_name, args.hf_text_col, args.hf_label_col, split=args.hf_split)

    # Route tasks
    if args.task == "baseline":
        assert y is not None, "Baseline classification requires labels."
        task_baseline_tfidf(X, y, test_size=args.test_size, out_prefix=args.out_prefix)

    elif args.task == "classify":
        assert y is not None, "Classification requires labels."
        task_classify_with_embeddings(X, y, embedder=args.embedder, test_size=args.test_size, out_prefix=args.out_prefix)

    elif args.task == "cluster":
        task_cluster_with_embeddings(X, embedder=args.embedder, k=args.k, out_prefix=args.out_prefix)

if __name__ == "__main__":
    main()
