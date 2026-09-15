from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

# 1. Tải ảnh từ URL (hoặc bạn có thể dùng Image.open("duong_dan_anh.jpg"))
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 2. Khởi tạo Processor và Model từ Hugging Face
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# 3. Tiền xử lý ảnh và đưa qua mô hình
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 4. Hậu xử lý kết quả
# Trích xuất các dự đoán có độ tin cậy (confidence) > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# 5. In kết quả
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    class_name = model.config.id2label[label.item()]
    print(f"Phát hiện {class_name} với độ tin cậy {round(score.item(), 3)} tại tọa độ {box}")