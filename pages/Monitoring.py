# pages/Monitoring.py
# ======================================
import streamlit as st
import altair as alt
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import logging
import os, sys, subprocess

from services.mongo import db
from services.monitoring import load_logs
from simulation.simulate_chat_tests import run_tests

st.set_page_config(page_title="Monitoring & Tests"#, page_icon="📊"
                   , layout="wide")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "chatbot-monitoring.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("ml/models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)

monitoring_col = db["monitoring"]
test_results_col = db["test_results"]

def show_summary_metrics(df: pd.DataFrame):
    if df is None or df.empty:
        return
    if {"execution_time","model"}.issubset(df.columns):
        st.write(f"Avg GPT-3 time: {df.loc[df['model']=='gpt-3','execution_time'].mean():.3f}s")
        st.write(f"Avg BERT time: {df.loc[df['model']=='bert','execution_time'].mean():.3f}s")
    if {"correct","model"}.issubset(df.columns) and df["correct"].dtype in [bool, int, float]:
        st.write(f"Overall Accuracy: {df['correct'].mean():.2%}")

st.title("📈 Monitoring & Automated Tests")

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Train Models", "🔍 Monitoring Logs", "🧪 Test Results", "🗂️ Execution Metrics"
])

# --- TAB 1: Train Models
with tab1:
    st.subheader("📤 Upload CSV & Train Models")
    uploaded = st.file_uploader("Upload CSV (customer_support_tickets-like)", type=["csv"])

    if uploaded:
        csv_path = DATA_DIR / uploaded.name
        csv_path.write_bytes(uploaded.getbuffer())
        st.success(f"Saved: {csv_path}")
        df_preview = pd.read_csv(csv_path)
        st.dataframe(df_preview.head())

        total = len(df_preview)
        info = st.info(f"📊 {total} rows detected.")
        pg = st.progress(0)
        for i in range(min(total, 50)):  # animação leve
            time.sleep(0.01)
            pg.progress(int((i+1)/max(1, min(total,50))*100))
        pg.empty(); info.empty()

        if st.button("🚀 Train models now"):
            with st.spinner("Training with run_evaluation.py ..."):
                cmd = [
                    sys.executable, "run_evaluation.py",
                    "--raw_csv", str(csv_path),
                    "--task", "both", "--cv", "0",
                    "--ng_min", "1", "--ng_max", "2",
                    "--C", "2.0", "--max_features", "100000"
                ]
                res = subprocess.run(cmd, check=False, capture_output=True, text=True)
                st.code(res.stdout or "", language="bash")
                if res.returncode == 0:
                    monitoring_col.insert_one({
                        "event": "train_models",
                        "file": str(csv_path),
                        "timestamp": datetime.now(timezone.utc),
                        "status": "success",
                        "log_source": "production"
                    })
                    st.success("✅ Models saved to ml/models/*_pipeline.joblib")
                else:
                    monitoring_col.insert_one({
                        "event": "train_models",
                        "file": str(csv_path),
                        "timestamp": datetime.now(timezone.utc),
                        "status": "error",
                        "error": res.stderr,
                        "log_source": "production"
                    })
                    st.error(res.stderr or "Unknown error")

# --- TAB 2: Monitoring Logs
with tab2:
    st.markdown("### 🔍 Monitoring Logs — Filter & Export")
    collections = {"Monitoring (prod)": "monitoring", "Test Results (simulation)": "test_results"}
    selected = st.selectbox("📂 Select the Collection", list(collections.keys()), index=0)
    col = db[collections[selected]]

    c1,c2,c3,c4 = st.columns(4)
    with c1: user_filter = st.text_input("Filter user_id (regex)")
    with c2: model_filter = st.text_input("Filter model (exact)")
    with c3: date_from = st.date_input("From", value=None)
    with c4: date_to = st.date_input("To", value=None)

    query = {}
    if user_filter: query["user_id"] = {"$regex": user_filter, "$options":"i"}
    if model_filter: query["model"] = model_filter
    if date_from:
        query.setdefault("timestamp", {})["$gte"] = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    if date_to:
        query.setdefault("timestamp", {})["$lte"] = datetime.combine(date_to, datetime.max.time(), tzinfo=timezone.utc)

    logs = list(col.find(query).sort("timestamp", -1).limit(5000))
    st.write(f"Showing {len(logs)} rows from **{selected}**.")
    if logs:
        df_logs = pd.DataFrame(logs)
        if "_id" in df_logs: df_logs["_id"] = df_logs["_id"].astype(str)
        if "timestamp" in df_logs: df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
        if "details" in df_logs: df_logs["details"] = df_logs["details"].astype(str)
        st.dataframe(df_logs.head(400))

        csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export CSV", data=csv_bytes, file_name="monitoring_logs.csv", mime="text/csv")
    else:
        st.info("No logs match the filter.")

# --- TAB 3: Test Results
with tab3:
    st.subheader("🧪 Chatbot Test Results")
    uploaded_questions = st.file_uploader("Upload questions (.txt) to run the test", type=["txt"])
    if uploaded_questions and st.button("🚀 Run Tests"):
        q_path = DATA_DIR / uploaded_questions.name
        q_path.write_bytes(uploaded_questions.getbuffer())
        qs = [ln.strip() for ln in q_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        st.info(f"📊 {len(qs)} questions loaded.")

        # sim animation
        pg = st.progress(0); box = st.empty()
        for i in range(len(qs)):
            time.sleep(0.01)
            pg.progress(int((i+1)/len(qs)*100))
        pg.empty(); box.empty()

        results = run_tests(str(q_path))
        df_up = pd.DataFrame(results)
        if "_id" in df_up: df_up["_id"] = df_up["_id"].astype(str)
        if "timestamp" in df_up: df_up["timestamp"] = pd.to_datetime(df_up["timestamp"])
        st.dataframe(df_up)

    model_options = ["All"] + sorted(test_results_col.distinct("model", {"log_source": "simulation"}))
    model_filter = st.selectbox("Filter by model", options=model_options, index=0)
    q = {"log_source": "simulation"}
    if model_filter != "All": q["model"] = model_filter

    results = list(test_results_col.find(q).sort("timestamp", -1))
    df = pd.DataFrame(results) if results else pd.DataFrame()
    if not df.empty:
        if "_id" in df: df["_id"] = df["_id"].astype(str)
        if "timestamp" in df: df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.dataframe(df)

        st.markdown("### 📊 Summary Metrics")
        show_summary_metrics(df)

        if {"model","execution_time"}.issubset(df.columns):
            chart = alt.Chart(df).mark_boxplot().encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("execution_time:Q", title="Response Time (s)"),
                color="model:N"
            ).properties(title="Response Time Distribution")
            st.altair_chart(chart, use_container_width=True)

        if {"model","correct"}.issubset(df.columns) and df["correct"].dtype in [bool, int, float]:
            df_acc = df.groupby("model")["correct"].mean().reset_index()
            chart_acc = alt.Chart(df_acc).mark_bar().encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("correct:Q", title="Accuracy"),
                color="model:N"
            ).properties(title="Accuracy by Model")
            st.altair_chart(chart_acc, use_container_width=True)

        confirm_delete = st.checkbox("Delete ALL test results?")
        if confirm_delete and st.button("🗑️ Confirm Delete"):
            res = test_results_col.delete_many(q)
            st.success(f"Deleted {res.deleted_count} documents.")
            st.rerun()
    else:
        st.info("No test results to display.")

# --- TAB 4: Execution Metrics
with tab4:
    st.subheader("🗂️ Execution Metrics Overview")
    log_type = st.selectbox("Select log type:", ["all", "train", "test"])
    dfm = load_logs(log_type)
    if not dfm.empty:
        st.dataframe(dfm)
        st.markdown("### 📊 Summary Metrics")
        show_summary_metrics(dfm)
    else:
        st.warning("No logs found for the selected type.")
