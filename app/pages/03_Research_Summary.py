import streamlit as st

st.set_page_config(page_title="Research Summary", layout="wide")
st.title("Research Summary")

st.markdown("""
## Project Title
Video-Based Deep Learning for Assessing Collaborative Learning Quality

## Problem Statement
Group learning does not automatically guarantee effective collaboration. This system analyzes temporal patterns in classroom group-discussion videos using CNN feature extraction and temporal models (SimpleRNN, LSTM, GRU) compared against a CNN baseline.

## Dataset
Primary candidate: **OUC-CGE** (Open University of China Classroom Group Engagement).  
Sources: [GitHub](https://github.com/oucyy361/OUC-CGE), [OSF](https://osf.io/brd2c/), [Nature Sci Data](https://www.nature.com/articles/s41597-025-04987-w)  
Labels (if available): Low, Medium, High (3-class). Actual class names must be verified from dataset.

## Methodology
- Video loading/validation, frame sampling
- CNN feature extraction (MobileNetV2 pretrained on ImageNet) as features
- CNN baseline on features
- Temporal classifiers on ordered feature sequences
- Reproducible splits, stratification where possible, no test leakage
- Metrics: Accuracy, Macro Precision/Recall/F1, per-class, confusion matrix

## Limitations (Key)
- Predictions are about group-level visual patterns, not definitive assessments of individuals
- No claims of detecting emotions/intelligence/motivation/intentions
- Engagement labels depend on dataset annotations
- Without learning outcome data, no outcome correlations claimed
- Temporal benefits not assumed - must be shown by evidence
- Dataset access may require permissions; components built independently

## Future Work
- Validate on actual OUC-CGE with official splits
- Audio/pose integration if data available
- Stronger temporal models with more data
- Explainability (attention) where justified
""")
