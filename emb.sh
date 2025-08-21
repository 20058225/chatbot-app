#!/bin/bash
clear

source ./myenv/bin/activate

echo "🚀 Running 'HF' tests..."
python -m services.embeddings --dataset hf

echo "@@ ========================================== @@"
echo "🚀 Running 'local' tests..."
python -m services.embeddings --dataset local

echo "@@ ========================================== @@"
echo "🚀 Running 'questions' tests..."
python -m services.embeddings --dataset questions

echo "@@ ========================================== @@"
echo "@@ ========================================== @@"
echo "🚀 Simulate chat sessions and log to Mongo..."
python -m simulation.simulate_chat_tests
