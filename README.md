
# 🤖 AI Chatbot Suite

A Streamlit-based AI Chatbot application with support for sentiment and priority classification of helpdesk tickets, FAQ import, and MongoDB integration. Includes training pipelines using machine learning models stored locally.

---

## 🚀 Features

- 📥 Import FAQs, default messages, or knowledge articles via CSV, JSON, or manual form.
- 🧠 Chatbot interface powered by OpenAI (via API key).
- 🧪 Sentiment & priority detection using pre-trained ML models (`joblib` pipelines).
- 📊 Admin dashboard for chat insights.
- 🐳 Docker & GitHub Actions integration for CI/CD.

---

## 🧰 Technologies

- Python 3.11
- Streamlit  
- MongoDB (via `pymongo`)  
- scikit-learn, joblib  
- Hugging Face `datasets` (optional, for evaluation)  
- Docker, GitHub Actions  

---

## 📂 Project Structure
```
.
├── ml/
│   └── models/
│       ├── priority\_pipeline.joblib
│       └── sentiment\_pipeline.joblib
├── data/
│   └── customer\_support\_tickets.csv
├── pages/
│   ├── Chatbot.py
│   ├── Dashboard.py
│   └── Monitoring.py
├── services/
│   ├── data\_loader.py
│   ├── evaluation.py
│   ├── import\_file.py
│   ├── ml.py
│   ├── mongo.py
│   ├── monitoring.py
│   └── schema\_detect.py
├── simulation/
│   ├── simulate\_chat\_tests.py
│   └── simulate\_eval\_emb.py
├── logs/
│   ├── chatbot-<timestamp>.log
│   ├── chatbot\_monitor.log
│   └── simulate\_chat\_tests.log
├── config/
│   ├── .env
│   └── .env.example
├── assets/
│   └── \[images/screenshots]
├── .github/
│   └── workflows/
│       └── chatbot-app.yml
├── Home.py
├── Dockerfile
├── docker.sh
├── emb.sh
├── run\_evaluation.py
├── requirements.txt
├── README.md
└── .gitignore
```
---

## ⚙️ Setup Instructions

### 🔧 Local (with Virtual Environment)

```bash
# Clone repo and move into folder
git clone https://github.com/20058225/chatbot-app.git
cd chatbot-app

# Create venv and activate it
python3 -m venv myenv
source myenv/bin/activate   # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py
````

Or simply:

```bash
bash docker.sh local
```

---

## 🐳 Run with Docker

```bash
# Build Docker image
docker build -t chatbot-app .

# Run container
docker run -p 8501:8501 --env-file=config/.env chatbot-app
```

---

## 🔐 Environment Variables (.env)

Create a file `config/.env` with the following content:

```
# OpenAI API key
OPENAI_API_KEY=your-openai-api-key

# MongoDB connection
API_MONGO=mongodb+srv://user:password@host/dbname

# Admin email (used for monitoring & notifications)
EMAIL_ADMIN=admin@example.com
EMAIL_PASS=your-email-app-password

# Docker Hub credentials (used only in GitHub Actions)
DOCKER_USERNAME=your-docker-username
DOCKER_PASSWORD=your-docker-password

# Render deploy hook (optional, used only if deploying via Render)
RENDER_API_KEY=render_api_key_here
RENDER_DEPLOY_HOOK=https://api.render.com/deploy/srv-xxxxxxxxxxxxxxxxxxxx
```

⚠️ **Important**:

* Local/Docker → only `OPENAI_API_KEY`, `API_MONGO`, `EMAIL_ADMIN`, `EMAIL_PASS` are required.
* GitHub Actions (CI/CD) → also set `DOCKER_USERNAME`, `DOCKER_PASSWORD`, `RENDER_API_KEY` and `RENDER_DEPLOY_HOOK` as **Secrets** in your repository.
* Never commit real credentials to Git.

---

## ✅ Run Tests

```bash
pytest tests/
```

---

## 🧠 ML Models

```
ml/models/sentiment_pipeline.joblib  → Sentiment classifier (positive/neutral/negative)

ml/models/priority_pipeline.joblib   → Ticket priority classifier (High/Medium/Low)
```

Both are loaded at startup by `services/ml.py`.

---

## 🗃️ Sample Data Format (customer\_support\_tickets.csv)

This dataset is used to train and evaluate both ML pipelines:

* **Sentiment classification** → `positive`, `neutral`, `negative`
* **Priority classification** → `High`, `Medium`, `Low`

### Example rows:

```csv
description,sentiment,priority
"I can't log in to my account",negative,High
"My computer is running slow",negative,Medium
"How do I reset my email password?",neutral,Medium
"The printer is showing a paper jam error",negative,Medium
"Request for software upgrade",neutral,Low
"Thank you for fixing my internet issue",positive,Low
```

---

## 🔄 CI/CD with GitHub Actions

* Runs on push/pull-request to `main`
* Installs dependencies and runs pytest
* Builds Docker image and (optionally) pushes to Docker Hub
* Triggers deploy on Render if `RENDER_DEPLOY_HOOK` is set

See [`.github/workflows/chatbot-app.yml`](.github/workflows/chatbot-app.yml).

---

## 📄 License

MIT © 2025 Brenda Lopes — [LICENSE](./LICENSE)

---

## ✨ Screenshots

### 🤖 Home Interface

![Home](assets/home.png)

### 🤖 Terminal

![Terminal](assets/terminal.png)

### 🤖 Chatbot Interface

![Chatbot 1](assets/chatbot_1.png)
![Chatbot 2](assets/chatbot_2.png)

### 📊 Admin Dashboard

![FAQs](assets/dashboard_faqs.png)
![Articles](assets/dashboard_articles.png)

### 📥 FAQ & Default Message Import

![Import 1](assets/import_1.png)
![Import 2](assets/import_2.png)

---

## 🙋‍♀️ About

This project was developed as part of the MSc in Computing & Information Systems at Dublin Business School.
The goal is to improve ticket triage using NLP, ML and a chatbot interface.

```
