"""
Pipeline chính: chạy toàn bộ bước a → b → c.
"""
import numpy as np
import pandas as pd
from keras.utils import to_categorical

from data_loader import (
    download_dataset, load_dataframe, explore_data,
    encode_labels, split_data
)
from preprocess import tokenize, build_vocab, encode_and_pad
from model import build_rnn_model
from train import train_model, plot_history
from evaluate import evaluate_model

from config import MAX_LEN, RANDOM_SEED


def main():
    # ==========================================
    # BƯỚC A: TẢI + EDA + CHIA TẬP
    # ==========================================
    print("\n" + "=" * 60)
    print("BƯỚC A: TẢI VÀ KHÁM PHÁ DỮ LIỆU")
    print("=" * 60)

    csv_path = download_dataset(force_download=False)
    df = load_dataframe(csv_path)
    explore_data(df)
    df, label_encoder = encode_labels(df)

    class_names = label_encoder.classes_.tolist()
    num_classes = len(class_names)

    (X_train_text, y_train), (X_val_text, y_val), (X_test_text, y_test) = \
        split_data(df)

    # ==========================================
    # TIỀN XỬ LÝ VĂN BẢN
    # ==========================================
    print("\n" + "=" * 60)
    print("TIỀN XỬ LÝ VĂN BẢN")
    print("=" * 60)

    # Tokenize
    train_tokens = [tokenize(t) for t in X_train_text]
    val_tokens   = [tokenize(t) for t in X_val_text]
    test_tokens  = [tokenize(t) for t in X_test_text]

    # Build vocab (CHỈ từ tập train để tránh data leakage)
    word2idx, idx2word = build_vocab(train_tokens)
    VOCAB_SIZE = len(word2idx)

    # Encode + Pad
    X_train = encode_and_pad(X_train_text, word2idx, MAX_LEN)
    X_val   = encode_and_pad(X_val_text,   word2idx, MAX_LEN)
    X_test  = encode_and_pad(X_test_text,  word2idx, MAX_LEN)

    print(f"\n📐 Shape X_train: {X_train.shape}")
    print(f"📐 Shape X_val  : {X_val.shape}")
    print(f"📐 Shape X_test : {X_test.shape}")

    # ==========================================
    # BƯỚC B: XÂY DỰNG MÔ HÌNH
    # ==========================================
    print("\n" + "=" * 60)
    print("BƯỚC B: XÂY DỰNG MÔ HÌNH RNN")
    print("=" * 60)

    model = build_rnn_model(
        vocab_size=VOCAB_SIZE,
        num_classes=num_classes,
        max_len=MAX_LEN,
        rnn_type="lstm",          # đổi thành 'gru' nếu muốn
        bidirectional=False,      # đổi thành True để dùng BiLSTM
    )

    # ==========================================
    # BƯỚC C: HUẤN LUYỆN + ĐÁNH GIÁ
    # ==========================================
    print("\n" + "=" * 60)
    print("BƯỚC C: HUẤN LUYỆN VÀ ĐÁNH GIÁ")
    print("=" * 60)

    history, model_path = train_model(
        model, X_train, y_train, X_val, y_val,
        model_name="rnn_lstm",
    )

    plot_history(history)

    results = evaluate_model(
        model, X_test, y_test,
        class_names=class_names,
    )

    print("\n✅ HOÀN TẤT PIPELINE!")


if __name__ == "__main__":
    main()