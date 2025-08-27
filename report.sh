# criar novo ambiente só para experimentos
# python3 -m venv myenv_experiments

echo "ativar (Linux/Mac)"
# source myenv_experiments/bin/activate


# instalar deps
# pip install --upgrade pip
# 1) Deps (ideal: novo venv)
# pip install -r requirements_experiments.txt

echo "Cleaning..."
clear
echo "Starting experiments..."
# 2) Baseline (TF-IDF)
echo "Baseline (TF-IDF)"
python experiments.py --task baseline --dataset local \
  --csv-path data/customer_support_tickets.csv \
  --text-col "Ticket Description" --label-col "Ticket Priority" \
  --out-prefix exp_local_tfidf

# 3) BERT (HF)
echo "BERT (HF)"
python experiments.py --task classify --dataset local \
  --embedder bert \
  --csv-path data/customer_support_tickets.csv \
  --text-col "Ticket Description" --label-col "Ticket Priority" \
  --out-prefix exp_local_bert

# 4) SBERT (HF) — CSV local
echo "SBERT (HF) — CSV local"
python experiments.py --task classify --dataset local \
  --embedder sbert \
  --csv-path data/customer_support_tickets.csv \
  --text-col "Ticket Description" --label-col "Ticket Priority" \
  --out-prefix exp_local_sbert

# 5) GPT-3 embeddings (OpenAI)
echo "GPT-3 embeddings (OpenAI)"
python experiments.py --task classify --dataset local \
  --embedder gpt3 \
  --csv-path data/customer_support_tickets.csv \
  --text-col "Ticket Description" --label-col "Ticket Priority" \
  --out-prefix exp_local_gpt3

# 6) (Opcional) Clustering com BERT, k=3
echo "(Opcional) Clustering com BERT, k=3"
python experiments.py --task cluster --dataset local \
  --embedder bert --k 3 \
  --csv-path data/customer_support_tickets.csv \
  --text-col "Ticket Description" \
  --out-prefix exp_local_bertclu

echo "desativar (Linux/Mac)"
deactivate