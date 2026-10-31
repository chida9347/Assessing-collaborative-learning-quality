import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")
st.markdown("Comparison of baseline and temporal models based on actual evaluation results.")

comp_path = Path("results/metrics/model_comparison.json")
if comp_path.exists():
    with open(comp_path) as f:
        data = json.load(f)
    st.json(data)
else:
    st.info("No model comparison results found yet. Run experiments to generate results.")

# Also look for individual metric files
metrics_dir = Path("results/metrics")
if metrics_dir.exists():
    files = list(metrics_dir.glob("*.json"))
    if files:
        st.subheader("Individual metric files")
        for f in files:
            st.caption(f.name)
    else:
        st.caption("No metric JSON files found.")
