# Methodology

## Pipeline
1. Video loading & validation (OpenCV)
2. Frame sampling (uniform over duration), resize to 224x224, RGB
3. Preprocessing via MobileNetV2 preprocess_input
4. Feature extraction with MobileNetV2 (ImageNet, include_top=False, avg pooling) -> feature vector per frame
5. Baseline: aggregate/use features with MLP classifier
6. Temporal: sequence of features to SimpleRNN/LSTM/GRU
7. Training with early stopping/checkpoints, stratified splits where possible
8. Evaluation: accuracy, macro precision/recall/F1, per-class, confusion matrix

## Design choices
- CPU-friendly, feature caching
- Reproducible seeds
- No test leakage
- Conservative claims

## Ablation
Planned: frame count (e.g., 4, 8, 16) with fixed split/protocol
