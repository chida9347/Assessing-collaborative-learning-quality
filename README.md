# Video-Based Deep Learning for Assessing Collaborative Learning Quality

A research-oriented deep learning framework for analyzing temporal patterns in classroom group-discussion videos to classify group engagement levels.

## Status
This project is under active development. See `docs/experiment_log.md` for current progress and `docs/limitations.md` for known constraints.

## Overview
- Title: Video-Based Deep Learning for Assessing Collaborative Learning Quality
- Objective: Classify group engagement using CNN-extracted features + temporal models (SimpleRNN/LSTM/GRU) with a CNN baseline
- Dataset (primary candidate): [OUC-CGE](https://github.com/oucyy361/OUC-CGE) ([OSF](https://osf.io/brd2c/), [Nature Sci Data](https://www.nature.com/articles/s41597-025-04987-w))

## Quickstart
1. Create a virtual environment (Python 3.11 recommended for TF compatibility; current env: Python 3.14 - verify)
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare dataset: see `data/README.md`
4. Run experiments or launch app: see instructions below

## Structure
See full structure in the repo. Key modules under `src/`, experiments under `experiments/`, app under `app/`.

## Installation
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
### Train baseline
```powershell
python experiments/train_baseline.py --config configs/config.yaml
```

### Train temporal models
```powershell
python experiments/train_temporal.py --config configs/config.yaml --model lstm
```

### Compare models
```powershell
python experiments/compare_models.py --config configs/config.yaml
```

### Run ablation
```powershell
python experiments/run_ablation.py --config configs/config.yaml
```

### Launch app
```powershell
streamlit run run_app.py
```

## Scientific Integrity
- Results only reported if actually obtained
- Limitations documented
- No fabricated metrics/labels/samples
- Distinction between implemented/planned/future

## License/Attribution
Follow dataset terms from OUC-CGE repository/OSF.
