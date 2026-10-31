import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.title("About")

st.markdown("""
## Application
Video-Based Deep Learning for Assessing Collaborative Learning Quality - a research prototype for group engagement analysis from classroom videos.

## Pipeline
1. Video validation & metadata
2. Frame extraction (configurable sampling)
3. Preprocessing (MobileNetV2)
4. Feature extraction (CNN)
5. Baseline (feature MLP) vs Temporal (RNN/LSTM/GRU)
6. Evaluation & visualization
7. Inference on uploaded videos

## Dataset Attribution
OUC-CGE dataset by OUC researchers. See [repository](https://github.com/oucyy361/OUC-CGE), [OSF](https://osf.io/brd2c/), and associated publication. Users must comply with dataset terms.

## Responsible Use
This system analyzes visual patterns only. It is not a definitive assessment tool for individuals and makes no claims about emotions, intelligence, or learning outcomes without supporting data.
""")
