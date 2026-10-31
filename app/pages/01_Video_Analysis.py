import streamlit as st
import os
import tempfile
from pathlib import Path

from src.inference import VideoAnalyzer

st.set_page_config(page_title="Video Analysis", layout="wide")
st.title("Video Analysis")
st.markdown("Upload a classroom group-discussion video for engagement prediction.")

@st.cache_resource
def get_analyzer():
    try:
        return VideoAnalyzer()
    except Exception as e:
        st.error(f"Failed to initialize analyzer: {e}")
        return None

analyzer = get_analyzer()

uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "wmv"])
if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
    tfile.write(uploaded.read())
    tfile.close()
    st.video(tfile.name)
    
    if analyzer is None:
        st.error("Analyzer not available")
    else:
        available_models = list(analyzer.models.keys())
        if not available_models:
            st.warning("No trained models found. Train models first or add checkpoints to models/ directories.")
        else:
            model_name = st.selectbox("Select model", available_models)
            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    res = analyzer.analyze(tfile.name, model_name=model_name)
                if not res.get("valid"):
                    st.error(f"Analysis failed: {res.get('error')}")
                else:
                    st.success(f"Predicted: {res['predicted_class']}")
                    st.markdown("**Probabilities:**")
                    for k, v in res["probabilities"].items():
                        st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v:.3f}")
                    st.caption(res.get("note", ""))
                    st.json({"metadata": res["metadata"], "model_name": res["model_name"]})
    
    try:
        os.unlink(tfile.name)
    except Exception:
        pass
else:
    st.info("Upload a video to begin analysis.")
