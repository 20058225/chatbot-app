# Home.py
# ======================================
import streamlit as st

st.set_page_config(page_title="AI Chatbot Suite", page_icon="🤖")
st.title("🤖 AI Chatbot Suite")

st.markdown("Use the menu below to navigate:")

st.page_link("pages/Chatbot.py", label="🧠 Chat with the AI Assistant")
st.page_link("pages/Dashboard.py", label="📊 Dashboard & Analytics")
st.page_link("pages/Monitoring.py", label="📈 Monitoring & Tests")

st.markdown("---")
st.caption("DBS • MSc Research • 2024/2025")
