import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc bức ảnh gốc và chuyển về dạng xám (Grayscale) để dễ quan sát nét
image_path = '/Users/tinnguyen/Workspace/IUH/15. DEEP LEARNING/deep-learning/Sample_Dog_Cat_Chicken/src/image_1549ca.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Không tìm thấy ảnh! Tín nhớ kiểm tra lại tên file và đường dẫn nhé.")
else:
    # 2. KHỞI TẠO FILTER (Khuôn mẫu 3x3)
    # Đây là bộ lọc Sobel chuyên phát hiện "Đường viền dọc" (Vertical Edges)
    vertical_filter = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # 3. BƯỚC CONVOLUTION (Tích chập)
    # cv2.filter2D sẽ lấy khuôn mẫu trên trượt qua toàn bộ điểm ảnh của img
    conv_output = cv2.filter2D(src=img, ddepth=cv2.CV_64F, kernel=vertical_filter)

    # 4. BƯỚC RELU (Dọn dẹp)
    # Hàm np.maximum(0, x) ép toàn bộ các số âm về 0
    relu_output = np.maximum(0, conv_output)

    # 5. TRỰC QUAN HÓA KẾT QUẢ
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("1. Ảnh gốc (Đầu vào)")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2. Sau Convolution (Filter trượt)")
    plt.imshow(conv_output, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("3. Sau ReLU (Dọn dẹp số âm)")
    plt.imshow(relu_output, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()