import torch
import cv2
import time
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

def main():
    # Tối ưu hóa phần cứng: Kích hoạt MPS cho Apple Silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Faster R-CNN] Đang chạy trên thiết bị: {device}\n")

    print("Đang tải Faster R-CNN (ResNet-50 FPN)...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    labels_map = weights.meta["categories"]

    image_path = '/Users/tinnguyen/Workspace/IUH/15. DEEP LEARNING/deep-learning/Chuong2_Bai3_CNN/data/COCO/bus.jpg' 

    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không tìm thấy ảnh '{image_path}'.")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Lưu kích thước ảnh gốc để dùng cho bước hậu xử lý (Post-process)
    original_image_sizes = [(img_rgb.shape[0], img_rgb.shape[1])] 
    
    img_tensor = F.to_tensor(img_rgb).to(device)

    # Warm-up GPU/MPS
    with torch.no_grad():
        for _ in range(3):
            model([img_tensor])

    start_time = time.time()
    
    # =========================================================================
    # KIẾN TRÚC FASTER R-CNN CHI TIẾT (Tách lớp forward)
    # =========================================================================
    with torch.no_grad():
        print("="*55)
        print(" BÓC TÁCH KIẾN TRÚC FASTER R-CNN")
        print("="*55)
        
        # 1. TIỀN XỬ LÝ (Transform)
        # Resize, normalize và đưa ảnh vào cấu trúc ImageList mà model yêu cầu
        images, _ = model.transform([img_tensor], None)
        print(f"1. Ảnh sau Transform: {images.tensors.shape}") # [Batch, C, H, W]

        # 2. BACKBONE (Feature Extraction + FPN)
        # Trích xuất đặc trưng qua ResNet50 và Feature Pyramid Network
        features = model.backbone(images.tensors)
        print(f"2. Backbone xuất ra {len(features)} Feature Maps (FPN levels): {list(features.keys())}")

        # 3. RPN (Region Proposal Network)
        # Sinh ra các bounding box đề xuất (Proposals) từ Feature Maps
        proposals, _ = model.rpn(images, features, targets=None)
        print(f"3. RPN đề xuất: {proposals[0].shape[0]} vùng (Region Proposals)")

        # 4. ROI POOLING & CLASSIFICATION + REGRESSION (Nằm trong roi_heads)
        # - Lấy proposals cắt (crop) đặc trưng từ Feature Maps (RoI Align).
        # - Chạy qua Fully Connected layers để phân loại (Class) và tinh chỉnh tọa độ (Box Regression).
        detections, _ = model.roi_heads(features, proposals, images.image_sizes, targets=None)
        print(f"4. RoI Heads (Pooling + FC) lọc lại còn: {detections[0]['boxes'].shape[0]} đối tượng")

        # 5. HẬU XỬ LÝ (Post-processing)
        # Scale tọa độ bounding box từ ảnh transform về lại kích thước ảnh gốc ban đầu
        final_preds = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)[0]
    # =========================================================================

    end_time = time.time()
    inference_time = end_time - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0

    # Hiển thị Bounding Box
    annotated_img = img_rgb.copy()
    for box, label_idx, score in zip(final_preds['boxes'], final_preds['labels'], final_preds['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            label_name = labels_map[label_idx]
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_img, f"{label_name} {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    print("\n" + "="*45)
    print(" KẾT QUẢ ĐÁNH GIÁ: FASTER R-CNN (BÓC TÁCH)")
    print("="*45)
    print(f"Thời gian suy luận (1 ảnh): {inference_time:.4f} giây")
    print(f"Tốc độ khung hình (FPS):    {fps:.2f}\n")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(annotated_img)
    plt.title(f"Faster R-CNN Architecture Unpacked | FPS: {fps:.2f}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()