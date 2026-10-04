"""
Bước c: Huấn luyện mô hình.
"""
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

from config import (
    EPOCHS, BATCH_SIZE, PATIENCE,
    MODEL_DIR, OUTPUT_DIR
)


def train_model(model, X_train, y_train, X_val, y_val,
                model_name: str = "rnn_lstm"):
    """Huấn luyện mô hình với EarlyStopping và ModelCheckpoint."""
    model_path = MODEL_DIR / f"{model_name}_best.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\n🚀 Bắt đầu huấn luyện ({EPOCHS} epochs, batch_size={BATCH_SIZE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n💾 Model tốt nhất đã lưu tại: {model_path}")
    return history, model_path


def plot_history(history, save: bool = True):
    """Vẽ biểu đồ loss & accuracy theo epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss theo epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy theo epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save:
        save_path = OUTPUT_DIR / "training_history.png"
        plt.savefig(save_path, dpi=150)
        print(f"💾 Đã lưu biểu đồ training: {save_path}")
    plt.show()