import os
import tensorflow as tf

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TƯƠNG ĐỐI
# ==========================================
# Vì file này nằm trong 'src', ta cần lấy đường dẫn ngược ra thư mục gốc
# sau đó trỏ vào thư mục 'data/train' và 'data/test'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')

# ==========================================
# 2. CẤU HÌNH THÔNG SỐ HYPERPARAMETERS
# ==========================================
# Kích thước ảnh chuẩn hóa (resize mọi ảnh về 128x128)
IMG_SIZE = (128, 128)

# VÌ DỮ LIỆU RẤT NHỎ (tổng cộng 60 ảnh Train), ta PHẢI hạ Batch Size xuống thấp.
# Nếu để 32, mỗi Epoch mô hình chỉ học có 2 lần là hết ảnh. Để 8 thì mô hình
# sẽ có 7-8 lần cập nhật trọng số trong mỗi Epoch, giúp nó hội tụ tốt hơn.
BATCH_SIZE = 8 

# ==========================================
# 3. HÀM TẠO ĐƯỜNG ỐNG NẠP DỮ LIỆU
# ==========================================
def load_datasets():
    print("--- ĐANG NẠP TẬP HUẤN LUYỆN (TRAIN) ---")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels='inferred',          # Tự động lấy tên thư mục (bear, bull, chicken) làm nhãn
        label_mode='categorical',   # Trả về mảng One-hot (ví dụ: [1, 0, 0])
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        crop_to_aspect_ratio=True,  # Thêm dòng này để không làm biến dạng gấu
        shuffle=True                # Xáo trộn ảnh để tránh học vẹt theo thứ tự
    )

    print("\n--- ĐANG NẠP TẬP KIỂM THỬ (TEST) ---")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        crop_to_aspect_ratio=True,  # Thêm dòng này để không làm biến dạng gấu
        shuffle=False               # Test thì không cần xáo trộn
    )
    
    return train_dataset, test_dataset

# ==========================================
# 4. KHỐI KIỂM THỬ ĐỘC LẬP (TESTING BLOCK)
# ==========================================
# Khối lệnh này chỉ chạy khi em thực thi trực tiếp file data_loader.py
if __name__ == "__main__":
    train_ds, test_ds = load_datasets()
    
    # Lấy ra danh sách các nhãn mà Keras tự nhận diện được
    class_names = train_ds.class_names
    print(f"\n[THÀNH CÔNG] Keras đã tìm thấy các nhãn: {class_names}")
    
    # Kiểm tra kích thước của 1 Batch (1 Lô dữ liệu)
    for images, labels in train_ds.take(1):
        print(f"Kích thước 1 lô ảnh (Batch Shape): {images.shape}")
        print(f"Kích thước 1 lô nhãn (Label Shape): {labels.shape}")