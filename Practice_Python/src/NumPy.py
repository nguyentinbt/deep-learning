# Import thư viện và đặt tên viết tắt
import numpy as np

# Input: Chuyển list thành numpy array
y_true = np.array([10,20,30])
y_pred = np.array([7,22,29])

# Tính sai số tuyệt đối cho toàn bộ mảng cùng lúc
"""
Cách tính sai số tuyệt đối bình thường: abs(y_true-y_pred)
"""

errors = np.abs(y_true-y_pred)
# => Đầu ra của error: [a, b, c]

print("Danh sách sai số tuyệt đối: ", errors)

# tính trung bình của tất cả các sai số

mea = np.mean(errors)
print("Sai số trung bình của cả tập dữ liệu MAE :", mea)

# Hàm tính sai số bình phương trung bình

sai_so = y_true-y_pred
mse = sai_so ** 2


