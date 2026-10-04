import torch
import torch.nn as nn


# 1. Khai bao mo hinh ANN: Tạo 1 lớp kết nối 2 đầu vào và 1 đầu ra
layer = nn.Linear(in_features=2, out_features=1)

#2. Đầu vào: dữ liệu khách hàng (Tuỏi 35, Thu nhập = 2000.0)
x= torch.tensor([[35.0, 2000.0]])
print(x)

#3. Xử lý: Đưa dữ liệu chạy qua lớp mạng
predict = layer(x)

#4. Đầu ra: là con số ngẫu nhiêm mà AI tính toán
print("Dự đoán cảu AI: ", predict)
print("Shape đầu ra: ", predict.shape)