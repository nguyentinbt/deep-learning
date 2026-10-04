import numpy as np

# Tạo mảng thực tế

y_true = np.array([1, 0, 1])
y_pred= np.array([0.8, 0.2, 0.0])

# Tinh sai so
sai_so = y_true-y_pred
print("Mang chua sai so: ", sai_so)

# tính bình phương sai số
binh_phuong_sai_so = sai_so ** 2
print("Gia tri binh phương sai so: ", binh_phuong_sai_so)

# Tính trung binh binh phuong sai so băng cach gọi ham mean
mean_square_error = np.mean(binh_phuong_sai_so)
print("Trung binh binh phuong sai so: ", mean_square_error)
