import cv2
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

def main():
    # Tối ưu hóa phần cứng: Kích hoạt MPS cho Apple Silicon
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[YOLO] Đang chạy trên thiết bị: {device}")

    print("Đang tải YOLOv8m...")
    model = YOLO('yolov8m.pt')

    image_path = '/Users/tinnguyen/Workspace/IUH/15. DEEP LEARNING/deep-learning/Chuong2_Bai3_CNN/data/COCO/bus.jpg'
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không tìm thấy ảnh '{image_path}'. Vui lòng thêm ảnh vào thư mục cùng cấp.")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Khởi động GPU (Warm-up)
    for _ in range(3):
        model.predict(img_rgb, device=device, verbose=False)

    # a) Dự đoán & c) Đo tốc độ suy luận (FPS)
    start_time = time.time()
    results = model.predict(img_rgb, device=device, verbose=False)
    end_time = time.time()

    inference_time = end_time - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0

    # b) Hiển thị Bounding Box (YOLO có sẵn hàm plot tiện lợi)
    annotated_img = results[0].plot()

    print("\n" + "="*45)
    print(" KẾT QUẢ ĐÁNH GIÁ: YOLOv8m")
    print("="*45)
    print(f"Thời gian suy luận (1 ảnh): {inference_time:.4f} giây")
    print(f"Tốc độ khung hình (FPS):    {fps:.2f}")
    print("\n[Chỉ số Benchmark trên tập COCO]")
    print("- Precision/Recall: F1-score xuất sắc nhờ cân bằng giữa 2 chỉ số.")
    print("- mAP (IoU 0.5:0.95): ~50.2%")
    print("- Ứng dụng: Tuyệt vời cho xử lý video trực tiếp (ví dụ: phát hiện té ngã).")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(annotated_img)
    plt.title(f"YOLOv8m | FPS: {fps:.2f}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
