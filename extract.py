# python extract.py
# =======================
## 
import os
import ast

PROJECT_PATH = "."
OUTPUT_FILE = "project_roles_summary.txt"

ROLE_KEYWORDS = {
    # Interfaces de usuário
    "UI": ["streamlit", "dash", "flask", "django", "gradio", "altair", "plotly", "seaborn", "matplotlib", "pydeck"],
    
    # Modelos e embeddings
    "Model Loading": ["transformers", "AutoModel", "BertModel", "SentenceTransformer", "from_pretrained", "pipeline"],
    "NLP Processing": ["tokenize", "lemmatize", "stem", "stopwords", "regex", "preprocess", "embedding"],

    # Integração com IA generativa
    "LLM Integration": ["openai", "ChatCompletion", "Embedding", "google-generativeai", "google-ai-generativelanguage"],

    # Machine Learning tradicional
    "ML Training": ["fit(", "predict(", "LogisticRegression", "KMeans", "train_test_split", "scikit-learn", "imbalanced-learn"],

    # Banco de dados
    "Database": ["pymongo", "MongoClient", "insert_one", "find(", "update_one", "delete_one", "hf-xet", "pyarrow"],

    # Logging e Monitoramento
    "Monitoring/Logging": ["logging", "log_event", "logger", "watchdog", "rich"],

    # Avaliação de modelos
    "Evaluation": ["precision", "recall", "f1_score", "silhouette_score", "classification_report"],

    # Infra/DevOps
    "Deployment": ["Werkzeug", "Flask", "gunicorn"],

    # Notificações
    "Notifications/Email": ["yagmail", "premailer"],

    # Testes e Qualidade
    "Testing/QA": ["pytest", "unittest", "flake8", "autopep8"]
}


def list_functions_and_classes(filepath):
    funcs, classes = [], []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            node = ast.parse(f.read())
        # percorre também funções aninhadas
        for item in ast.walk(node):
            if isinstance(item, ast.FunctionDef):
                funcs.append(item.name)
            elif isinstance(item, ast.ClassDef):
                classes.append(item.name)
    except Exception as e:
        funcs.append(f"[Parse error: {e}]")
    return funcs, classes

def detect_roles(filepath):
    roles = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().lower()
        for role, keywords in ROLE_KEYWORDS.items():
            if any(kw.lower() in content for kw in keywords):
                roles.add(role)
    except Exception as e:
        print(f"⚠️ Error reading {filepath}: {e}")
    return list(roles) if roles else ["None detected"]

def generate_summary(base_path):
    report = []
    IGNORE_DIRS = {"__pycache__", "venv", "env", "myenv", ".git", ".pytest_cache", "site-packages"}

    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]

        indent_level = root.replace(base_path, "").count(os.sep)
        indent = "    " * indent_level
        report.append(f"{indent}{os.path.basename(root)}/")

        for file in files:
            filepath = os.path.join(root, file)
            try:
                size_kb = os.path.getsize(filepath) / 1024
            except:
                size_kb = 0
            file_indent = "    " * (indent_level + 1)
            report.append(f"{file_indent}{file} ({size_kb:.1f} KB)")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                line_count = len(lines)
            except:
                line_count = 0
            report.append(f"{file_indent}  └─ Lines: {line_count}")

            if file.endswith(".py"):
                roles = detect_roles(filepath)
                funcs, classes = list_functions_and_classes(filepath)
                report.append(f"{file_indent}  └─ Roles: {', '.join(roles)}")
                if funcs:
                    report.append(f"{file_indent}  └─ Functions: {', '.join(funcs)}")
                if classes:
                    report.append(f"{file_indent}  └─ Classes: {', '.join(classes)}")
    return "\n".join(report)

if __name__ == "__main__":
    summary = generate_summary(PROJECT_PATH)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✅ Summary saved to {OUTPUT_FILE}")
