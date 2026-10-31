from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, SimpleRNN, LSTM, GRU, GlobalAveragePooling2D, Flatten
from tensorflow.keras.applications import MobileNetV2


def build_cnn_baseline_feature_model(feature_dim=1280, num_classes=3, dropout=0.3):
    # Baseline can be built on top of features; simpler MLP
    inputs = Input(shape=(feature_dim,))
    x = Dropout(dropout)(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


def build_temporal_rnn(feature_dim=1280, sequence_length=16, hidden_units=64, num_classes=3, dropout=0.3, bidirectional=False):
    inputs = Input(shape=(sequence_length, feature_dim))
    if bidirectional:
        from tensorflow.keras.layers import Bidirectional
        x = Bidirectional(SimpleRNN(hidden_units, return_sequences=False, dropout=dropout))(inputs)
    else:
        x = SimpleRNN(hidden_units, return_sequences=False, dropout=dropout)(inputs)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


def build_temporal_lstm(feature_dim=1280, sequence_length=16, hidden_units=128, num_classes=3, dropout=0.3, bidirectional=False):
    inputs = Input(shape=(sequence_length, feature_dim))
    if bidirectional:
        from tensorflow.keras.layers import Bidirectional
        x = Bidirectional(LSTM(hidden_units, return_sequences=False, dropout=dropout))(inputs)
    else:
        x = LSTM(hidden_units, return_sequences=False, dropout=dropout)(inputs)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


def build_temporal_gru(feature_dim=1280, sequence_length=16, hidden_units=128, num_classes=3, dropout=0.3, bidirectional=False):
    inputs = Input(shape=(sequence_length, feature_dim))
    if bidirectional:
        from tensorflow.keras.layers import Bidirectional
        x = Bidirectional(GRU(hidden_units, return_sequences=False, dropout=dropout))(inputs)
    else:
        x = GRU(hidden_units, return_sequences=False, dropout=dropout)(inputs)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model
