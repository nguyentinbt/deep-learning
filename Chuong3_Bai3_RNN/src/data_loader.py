"""
Bước a: Tải dữ liệu, EDA, chia tập train/val/test.
"""
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    DATA_DIR, KAGGLE_DATASET, CSV_FILENAME,
    TEXT_COL, LABEL_COL,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED, OUTPUT_DIR
)


def download_dataset(force_download: bool = False) -> Path:
    """Tải dataset từ Kaggle về thư mục Data/ và trả về đường dẫn file CSV."""
    print("⏳ Đang tải dataset từ Kaggle...")
    path = kagglehub.dataset_download(
        KAGGLE_DATASET,
        output_dir=str(DATA_DIR),
        force_download=force_download,
    )
    csv_path = Path(path) / CSV_FILENAME
    print(f"✅ Dataset đã tải về: {csv_path}")
    return csv_path


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Đọc file CSV thành DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"\n📊 Kích thước dữ liệu: {df.shape}")
    print(f"📋 Các cột: {df.columns.tolist()}")
    print(f"\n🔍 5 dòng đầu tiên:")
    print(df.head())
    return df


def explore_data(df: pd.DataFrame, save_plot: bool = True):
    """EDA: hiển thị mẫu, thống kê, vẽ phân phối nhãn."""
    print("\n" + "=" * 50)
    print("KHÁM PHÁ DỮ LIỆU")
    print("=" * 50)

    # 5 comment ngẫu nhiên
    print("\n🎲 5 comment ngẫu nhiên:")
    print(df.sample(5, random_state=RANDOM_SEED))

    # Phân phối nhãn
    print(f"\n📈 Phân phối nhãn (cột '{LABEL_COL}'):")
    label_counts = df[LABEL_COL].value_counts()
    print(label_counts)
    print("\nTỷ lệ %:")
    print((df[LABEL_COL].value_counts(normalize=True) * 100).round(2))

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=LABEL_COL, order=label_counts.index)
    plt.title("Phân phối các nhãn cảm xúc")
    plt.xlabel("Nhãn")
    plt.ylabel("Số lượng")
    plt.tight_layout()

    if save_plot:
        save_path = OUTPUT_DIR / "label_distribution.png"
        plt.savefig(save_path, dpi=150)
        print(f"\n💾 Đã lưu biểu đồ: {save_path}")

    plt.show()


def encode_labels(df: pd.DataFrame):
    """Mã hóa nhãn string → số nguyên."""
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df[LABEL_COL])
    print(f"\n🏷️  Các lớp: {le.classes_.tolist()}")
    print(f"🔢 Ánh xạ: {dict(zip(le.classes_, range(len(le.classes_))))}")
    return df, le


def split_data(df: pd.DataFrame):
    """Chia tập train / val / test theo tỷ lệ 70/15/15 (stratified)."""
    X = df[TEXT_COL].values
    y = df["label_encoded"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"\n✂️  Chia tập:")
    print(f"   Train      : {len(X_train)} mẫu")
    print(f"   Validation : {len(X_val)} mẫu")
    print(f"   Test       : {len(X_test)} mẫu")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    csv_path = download_dataset(force_download=False)
    df = load_dataframe(csv_path)
    explore_data(df)
    df, label_encoder = encode_labels(df)
    train, val, test = split_data(df)
    print("\n✅ Hoàn tất bước a!")