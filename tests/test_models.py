import numpy as np
from src.models import build_cnn_baseline_feature_model, build_temporal_rnn, build_temporal_lstm, build_temporal_gru


def test_baseline_shape():
    m = build_cnn_baseline_feature_model(feature_dim=1280, num_classes=3)
    x = np.random.randn(4, 1280).astype(np.float32)
    y = m.predict(x, verbose=0)
    assert y.shape == (4, 3)
    # probs sum to ~1
    assert np.allclose(y.sum(axis=1), 1.0)


def test_rnn_shape():
    m = build_temporal_rnn(feature_dim=1280, sequence_length=16, num_classes=3)
    x = np.random.randn(2, 16, 1280).astype(np.float32)
    y = m.predict(x, verbose=0)
    assert y.shape == (2, 3)


def test_lstm_shape():
    m = build_temporal_lstm(feature_dim=1280, sequence_length=16, num_classes=3)
    x = np.random.randn(2, 16, 1280).astype(np.float32)
    y = m.predict(x, verbose=0)
    assert y.shape == (2, 3)


def test_gru_shape():
    m = build_temporal_gru(feature_dim=1280, sequence_length=16, num_classes=3)
    x = np.random.randn(2, 16, 1280).astype(np.float32)
    y = m.predict(x, verbose=0)
    assert y.shape == (2, 3)
