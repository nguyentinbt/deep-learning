"""
Tiền xử lý văn bản: tokenize, build vocab, encode, padding.
"""
import numpy as np
from collections import Counter

from config import (
    MIN_FREQ, MAX_LEN, PAD_TOKEN, UNK_TOKEN
)


def tokenize(text: str) -> list:
    """Tách câu thành list các từ (lowercase)."""
    return str(text).lower().split()


def build_vocab(token_lists, min_freq: int = MIN_FREQ):
    """Xây dựng từ điển word → index."""
    all_words = [w for tokens in token_lists for w in tokens]
    word_counts = Counter(all_words)

    vocab = [w for w, c in word_counts.items() if c >= min_freq]

    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w in vocab:
        word2idx[w] = len(word2idx)

    idx2word = {i: w for w, i in word2idx.items()}
    print(f"📚 Kích thước vocabulary: {len(word2idx)}")
    return word2idx, idx2word


def encode(tokens: list, word2idx: dict) -> list:
    """Chuyển list từ → list index."""
    return [word2idx.get(w, word2idx[UNK_TOKEN]) for w in tokens]


def pad_sequence(seq: list, max_len: int = MAX_LEN, pad_idx: int = 0) -> list:
    """Đưa câu về cùng độ dài max_len."""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


def encode_and_pad(texts, word2idx: dict, max_len: int = MAX_LEN) -> np.ndarray:
    """Pipeline: text → tokenize → encode → pad → numpy array."""
    result = []
    for text in texts:
        tokens = tokenize(text)
        seq = encode(tokens, word2idx)
        padded = pad_sequence(seq, max_len)
        result.append(padded)
    return np.array(result, dtype=np.int32)