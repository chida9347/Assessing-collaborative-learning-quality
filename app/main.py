import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="Video-Based Deep Learning for Assessing Collaborative Learning Quality", layout="wide", initial_sidebar_state="expanded")

st.title("Video-Based Deep Learning for Assessing Collaborative Learning Quality")
st.markdown("""
**Research-oriented system for analyzing temporal patterns in classroom group-discussion videos.**

This application classifies group engagement using CNN-extracted visual features with temporal models (SimpleRNN, LSTM, GRU) and provides a CNN baseline for comparison.
""")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the pages in the sidebar to explore the system.")

# Quick stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Project", "OUC-CGE focused")
with col2:
    models_dir = Path("models")
    available = []
    for p in ["models/cnn/baseline_model.keras", "models/temporal/rnn_model.keras", "models/temporal/lstm_model.keras", "models/temporal/gru_model.keras"]:
        if Path(p).exists():
            available.append(Path(p).name)
    st.metric("Trained models found", len(available))
with col3:
    st.metric("Classes", "Low / Medium / High (verify)")

st.markdown("---")
st.subheader("Research Objectives")
st.markdown("""
1. Load and validate classroom group-discussion videos
2. Extract frames and visual features (CNN/MobileNetV2)
3. Build CNN baseline and temporal models (SimpleRNN/LSTM/GRU)
4. Train/evaluate with reproducible methodology
5. Compare models with appropriate metrics
6. Provide web interface for video analysis
7. Investigate temporal patterns where supported
8. Run at least one ablation experiment
9. Document limitations and results
""")

st.info("All statistics shown are from actual project state when available. See docs/ for details.")
