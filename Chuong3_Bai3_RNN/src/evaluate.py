"""
Bước c: Đánh giá mô hình trên tập test.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from config import OUTPUT_DIR


def evaluate_model(model, X_test, y_test, class_names=None,
                   save_plot: bool = True):
    """Đánh giá mô hình: Accuracy, Precision, Recall, F1, Confusion Matrix."""
    print("\n" + "=" * 50)
    print("ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 50)

    # Dự đoán
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n📊 Accuracy : {acc:.4f}")
    print(f"📊 Precision: {prec:.4f}")
    print(f"📊 Recall   : {rec:.4f}")
    print(f"📊 F1-score : {f1:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=class_names,
                                zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.tight_layout()

    if save_plot:
        save_path = OUTPUT_DIR / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150)
        print(f"💾 Đã lưu confusion matrix: {save_path}")

    plt.show()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }