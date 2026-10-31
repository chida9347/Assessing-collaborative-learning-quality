# Limitations

## Technical
- Python/TensorFlow environment compatibility (current env: Python 3.14) - may need older Python for TF stability
- Dataset may not be present locally; components work without it
- Training time depends on data/hardware
- Feature extraction is CPU-intensive without GPU

## Scientific
- Group-level visual analysis only; not individual assessment
- No claims about emotions, intelligence, motivation, intentions, or mental states
- Cannot infer speaking/contribution quality from movement alone
- Engagement labels are dataset-defined
- No learning outcome correlations without data
- Temporal models not guaranteed to outperform baseline

## Data
- Official splits preferred but may not be discoverable
- Class imbalance possible
- Small test sets limit confidence
- Session-level leakage possible if clips from same session not separated
