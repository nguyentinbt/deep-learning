import torch
import cv2
import time
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from ultralytics import YOLO

def calculate_iou(boxA, boxB):
    """Tính toán độ chồng lấp (Intersection over Union) giữa 2 bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Khởi động Pipeline Đánh giá Mô hình trên: {device}\n")

    # ==========================================
    # 1. CHUẨN BỊ ẢNH
    # ==========================================
    image_path = '/Users/tinnguyen/Workspace/IUH/15. DEEP LEARNING/deep-learning/Chuong2_Bai3_CNN/data/COCO/bus.jpg'
    # image_path = '/Users/tinnguyen/Workspace/IUH/15. DEEP LEARNING/deep-learning/Chuong2_Bai3_CNN/data/COCO/cats.png'
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Lỗi: Không tìm thấy ảnh '{image_path}'.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ==========================================
    # 2. CHẠY YOLOv8m
    # ==========================================
    print("⏳ Đang tải và chạy YOLOv8m...")
    yolo_model = YOLO('yolov8m.pt')
    
    # Warm-up
    for _ in range(3): yolo_model.predict(img_rgb, device=device.type, verbose=False)
    
    yolo_start = time.time()
    yolo_results = yolo_model.predict(img_rgb, device=device.type, verbose=False)
    yolo_time = time.time() - yolo_start
    
    yolo_annotated = yolo_results[0].plot()
    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy() # Trích xuất tọa độ box
    yolo_classes = yolo_results[0].boxes.cls.cpu().numpy()
    yolo_names = yolo_model.names

    # ==========================================
    # 3. CHẠY FASTER R-CNN (BÓC TÁCH KIẾN TRÚC)
    # ==========================================
    print("⏳ Đang tải và chạy Faster R-CNN (ResNet-50 FPN)...")
    rcnn_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    rcnn_model = fasterrcnn_resnet50_fpn(weights=rcnn_weights).to(device).eval()
    rcnn_labels_map = rcnn_weights.meta["categories"]
    
    img_tensor = F.to_tensor(img_rgb).to(device)
    original_size = [(img_rgb.shape[0], img_rgb.shape[1])]

    # Warm-up
    with torch.no_grad():
        for _ in range(3): rcnn_model([img_tensor])

    rcnn_start = time.time()
    with torch.no_grad():
        images, _ = rcnn_model.transform([img_tensor], None)
        features = rcnn_model.backbone(images.tensors)
        proposals, _ = rcnn_model.rpn(images, features, targets=None)
        detections, _ = rcnn_model.roi_heads(features, proposals, images.image_sizes, targets=None)
        rcnn_preds = rcnn_model.transform.postprocess(detections, images.image_sizes, original_size)[0]
    rcnn_time = time.time() - rcnn_start

    rcnn_annotated = img_rgb.copy()
    rcnn_boxes_valid = []
    rcnn_labels_valid = []

    # for box, label_idx, score in zip(rcnn_preds['boxes'], rcnn_preds['labels'], rcnn_preds['scores']):
    #     if score > 0.5:
    #         x1, y1, x2, y2 = map(int, box)
    #         label_name = rcnn_labels_map[label_idx]
    #         cv2.rectangle(rcnn_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #         cv2.putText(rcnn_annotated, f"{label_name} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    #         rcnn_boxes_valid.append(box.cpu().numpy())
    #         rcnn_labels_valid.append(label_name)

    for box, label_idx, score in zip(rcnn_preds['boxes'], rcnn_preds['labels'], rcnn_preds['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            label_name = rcnn_labels_map[label_idx]
            
            # Thiết lập màu Xanh lá (RGB: 0, 255, 0) cho box và Đen (0, 0, 0) cho chữ
            box_color = (0, 255, 0)
            text_color = (0, 0, 0) 
            
            # 1. Vẽ bounding box
            cv2.rectangle(rcnn_annotated, (x1, y1), (x2, y2), box_color, 2)
            
            # 2. Tính toán kích thước của đoạn text để vẽ khung nền
            text = f"{label_name} {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # 3. Vẽ khung nền cho text (dùng tham số -1 để tô kín hình chữ nhật)
            cv2.rectangle(rcnn_annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), box_color, -1)
            
            # 4. In chữ lên trên khung nền vừa vẽ
            cv2.putText(rcnn_annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            rcnn_boxes_valid.append(box.cpu().numpy())
            rcnn_labels_valid.append(label_name)

    # ==========================================
    # 4. SO SÁNH VÀ ĐÁNH GIÁ (IoU & mAP)
    # ==========================================
    print("\n" + "="*60)
    print(" 📊 BÁO CÁO SO SÁNH HIỆU SUẤT VÀ ĐỘ ĐỒNG THUẬN (IoU)")
    print("="*60)
    print(f"[1] TỐC ĐỘ (FPS):")
    print(f"    - YOLOv8m:      {1/yolo_time:.2f} FPS ({yolo_time:.4f}s)")
    print(f"    - Faster R-CNN: {1/rcnn_time:.2f} FPS ({rcnn_time:.4f}s)")
    
    print(f"\n[2] ĐỘ CHÍNH XÁC (Dựa trên Benchmark mAP COCO):")
    print(f"    - YOLOv8m mAP@0.5:0.95: ~50.2%")
    print(f"    - Faster R-CNN mAP:     ~37.0 - 41.0%")

    print(f"\n[3] ĐỘ CHỒNG LẤP (IoU) GIỮA 2 MÔ HÌNH TRÊN ẢNH NÀY:")
    print("    Tiến hành so khớp các Box của YOLO và Faster R-CNN...")
    
    match_count = 0
    for i, y_box in enumerate(yolo_boxes):
        y_label = yolo_names[int(yolo_classes[i])]
        best_iou = 0
        best_r_label = ""
        
        for j, r_box in enumerate(rcnn_boxes_valid):
            iou = calculate_iou(y_box, r_box)
            if iou > best_iou:
                best_iou = iou
                best_r_label = rcnn_labels_valid[j]
        
        if best_iou > 0.5: # Ngưỡng IoU > 0.5 được xem là khớp
            match_count += 1
            print(f"    ✅ [Khớp] {y_label} (YOLO) & {best_r_label} (R-CNN) | IoU = {best_iou:.2f}")
        elif best_iou > 0:
            print(f"    ⚠️ [Lệch] {y_label} (YOLO) & {best_r_label} (R-CNN) | IoU = {best_iou:.2f} (< 0.5)")
            
    print(f"\n    => Tổng kết: 2 mô hình đồng thuận trên {match_count}/{len(yolo_boxes)} đối tượng do YOLO phát hiện.")
    print("="*60)

    # ==========================================
    # 5. HIỂN THỊ VISUALIZATION
    # ==========================================
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(yolo_annotated)
    axs[0].set_title(f"YOLOv8m\nFPS: {1/yolo_time:.2f} | Boxes: {len(yolo_boxes)}")
    axs[0].axis('off')

    axs[1].imshow(rcnn_annotated)
    axs[1].set_title(f"Faster R-CNN (Bóc tách)\nFPS: {1/rcnn_time:.2f} | Boxes: {len(rcnn_boxes_valid)}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()