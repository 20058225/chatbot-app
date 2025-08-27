#!/usr/bin/env bash
set -euo pipefail

# Script para mapear a estrutura do projeto ignorando "myenv"
# Uso: ./project_tree.sh > project_structure.txt

ROOT_DIR=$(pwd)

echo "📂 Estrutura do projeto em: $ROOT_DIR"
echo

# Se tiver tree instalado → usar tree
if command -v tree &> /dev/null; then
    tree -I "myenv|__pycache__|*.pyc|*.pyo" -a
else
    echo "⚠️ 'tree' não encontrado. Usando 'find'."
    find . \
      -path "./myenv" -prune -o \
      -path "*/__pycache__" -prune -o \
      -name "*.pyc" -prune -o \
      -name "*.pyo" -prune -o \
      -name "*.git" -prune -o \
      -print
fi
