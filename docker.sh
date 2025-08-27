#!/usr/bin/env bash
set -e

MODE=${1:-docker}   # use: ./docker.sh       (docker)
                    #      ./docker.sh local (venv local)

mkdir -p logs

if [ "$MODE" = "local" ]; then
  source ./myenv/bin/activate
  streamlit run Home.py --server.headless true \
    2>&1 | tee "logs/chatbot-$(date +'%Y%m%d-%H%M%S').log"
else
  # builda e roda no Docker
  docker build -t chatbot-app .

  docker run -i --rm \
    -p 8501:8501 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/ml/models:/app/ml/models" \
    --env-file config/.env \
    chatbot-app \
    2>&1 | tee "logs/chatbot-$(date +'%Y%m%d-%H%M%S').log"
fi
