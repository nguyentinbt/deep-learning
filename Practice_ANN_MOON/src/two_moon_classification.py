import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# ==========================================
# KHỐI 1: TẠO DỮ LIỆU SẠCH (Chỉ 3 dòng code)
# ==========================================
# Sinh 1000 mẫu dữ liệu, noise=0.1 để các điểm hơi nhiễu một chút cho thực tế
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Chuẩn hóa nhanh và ép kiểu sang Tensor để PyTorch đọc được
X = StandardScaler().fit_transform(X)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).view(-1, 1) # Định hình lại nhãn thành cột dọc

# ==========================================
# KHỐI 2: KHAI BÁO MÔ HÌNH ANN
# ==========================================
class SimpleMoonANN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input có 2 đặc trưng (tọa độ X, Y), lớp ẩn có 16 nơ-ron
        self.hidden = nn.Linear(2, 16) 
        self.relu = nn.ReLU()
        # Output từ 16 nơ-ron về 1 (xác suất 0 đến 1)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# ==========================================
# KHỐI 3: KHỞI TẠO LUẬT CHƠI & VÒNG LẶP HUẤN LUYỆN
# ==========================================
model = SimpleMoonANN()
criterion = nn.BCELoss() # Sai số cho phân loại nhị phân (0 hoặc 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
print("Bắt đầu huấn luyện...")

for epoch in range(epochs):
    model.train()
    
    # 1. Dự đoán
    y_pred = model(X_tensor)
    
    # 2. Tính sai số
    loss = criterion(y_pred, y_tensor)
    
    # 3. Xóa đạo hàm cũ
    optimizer.zero_grad()
    
    # 4. Truyền ngược tính đạo hàm mới
    loss.backward()
    
    # 5. Cập nhật trọng số mạng
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Hoàn thành!")