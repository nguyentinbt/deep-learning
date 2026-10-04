"""
Định nghĩa mô hình RNN (LSTM / GRU / BiLSTM).
"""
from keras.models import Sequential
from keras.layers import (
    Embedding, LSTM, GRU, Dense, Dropout,
    SpatialDropout1D, Bidirectional
)
from keras.optimizers import Adam

from config import (
    EMBEDDING_DIM, HIDDEN_SIZE, DROPOUT_RATE,
    DENSE_UNITS, LEARNING_RATE
)


def build_rnn_model(vocab_size: int, num_classes: int, max_len: int,
                    rnn_type: str = "lstm", bidirectional: bool = False):
    """
    Xây dựng mô hình RNN.

    Args:
        vocab_size    : kích thước vocabulary
        num_classes   : số lớp output (3 cho bài này)
        max_len       : độ dài chuỗi input
        rnn_type      : 'lstm' hoặc 'gru'
        bidirectional : dùng Bidirectional hay không
    """
    # Chọn loại RNN
    if rnn_type.lower() == "lstm":
        rnn_layer = LSTM(HIDDEN_SIZE, dropout=DROPOUT_RATE)
    elif rnn_type.lower() == "gru":
        rnn_layer = GRU(HIDDEN_SIZE, dropout=DROPOUT_RATE)
    else:
        raise ValueError("rnn_type phải là 'lstm' hoặc 'gru'")

    if bidirectional:
        rnn_layer = Bidirectional(rnn_layer)

    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=EMBEDDING_DIM,
                  input_length=max_len,
                  name="embedding"),
        SpatialDropout1D(0.2, name="spatial_dropout"),
        rnn_layer,
        Dense(DENSE_UNITS, activation="relu", name="fc1"),
        Dropout(DROPOUT_RATE, name="dropout"),
        Dense(num_classes, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model