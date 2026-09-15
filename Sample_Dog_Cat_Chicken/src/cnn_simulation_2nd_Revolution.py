import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc bức ảnh gốc
image_path = '/Users/tinnguyen/Workspace/IUH/15. DEEP LEARNING/deep-learning/Sample_Dog_Cat_Chicken/src/image_1549ca.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Không tìm thấy ảnh! Tín nhớ kiểm tra lại tên file và đường dẫn nhé.")
else:
    # ==========================================
    # LỚP 1: TÌM ĐƯỜNG NÉT DỌC
    # ==========================================
    vertical_filter = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    conv1_output = cv2.filter2D(src=img, ddepth=cv2.CV_64F, kernel=vertical_filter)
    relu1_output = np.maximum(0, conv1_output)

    # ==========================================
    # BƯỚC CHUYỂN GIAO: MAX POOLING 2x2
    # ==========================================
    # Lấy kích thước ảnh sau ReLU 1. Cắt viền nếu kích thước bị lẻ (để chia hết cho 2)
    h, w = relu1_output.shape
    h_pool, w_pool = h // 2, w // 2
    
    # Dùng mẹo reshape của NumPy để chia ảnh thành các ô 2x2 và lấy giá trị Max
    pooled_output = relu1_output[:h_pool*2, :w_pool*2].reshape(h_pool, 2, w_pool, 2).max(axis=(1, 3))

    # ==========================================
    # LỚP 2: TÌM ĐẶC TRƯNG PHỨC TẠP HƠN
    # ==========================================
    # Lần này ta dùng bộ lọc ngang (Horizontal Filter) trượt trên ảnh đã pooling
    horizontal_filter = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    conv2_output = cv2.filter2D(src=pooled_output, ddepth=cv2.CV_64F, kernel=horizontal_filter)
    relu2_output = np.maximum(0, conv2_output)

    # ==========================================
    # TRỰC QUAN HÓA KẾT QUẢ (6 BƯỚC)
    # ==========================================
    plt.figure(figsize=(15, 8)) # Phóng to figure ra một chút để chứa 6 ảnh

    plt.subplot(2, 3, 1)
    plt.title("1. Ảnh gốc")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("2. Conv 1 (Tìm nét dọc)")
    plt.imshow(conv1_output, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("3. ReLU 1 (Dọn dẹp)")
    plt.imshow(relu1_output, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("4. Max Pooling 1 (Thu nhỏ)")
    plt.imshow(pooled_output, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("5. Conv 2 (Tìm nét ngang)")
    plt.imshow(conv2_output, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("6. ReLU 2 (Đặc trưng cấp 2)")
    plt.imshow(relu2_output, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()