# User Guide

## Installation
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare Data
1. Download OUC-CGE from official sources (GitHub/OSF)
2. Place videos in `data/raw/` - organize by class folders (`Low/`, `Medium/`, `High/`) if available
3. Create splits: can use `DatasetManager.create_splits()` or place official splits

## Extract Features & Train
Use experiment scripts; feature caching recommended.

## Run App
```powershell
streamlit run run_app.py
```
Or `streamlit run app/main.py`

## Testing
```powershell
pytest -v
```
