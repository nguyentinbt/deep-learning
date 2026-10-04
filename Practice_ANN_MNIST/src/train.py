import torch

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    for epoch in range(epochs):
        model.train() # Đặt model ở chế độ train[cite: 2]
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Đẩy dữ liệu lên GPU/CPU[cite: 2]
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Reset gradient tránh cộng dồn[cite: 2]
            optimizer.zero_grad()
            
            # 2. Forward pass[cite: 2]
            outputs = model(inputs)
            
            # 3. Tính Loss[cite: 2]
            loss = criterion(outputs, labels)
            
            # 4. Backward pass (Tính gradient)[cite: 2]
            loss.backward()
            
            # 5. Cập nhật trọng số[cite: 2]
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval() # Chuyển sang chế độ evaluation[cite: 2]
    correct = 0
    total = 0
    
    with torch.no_grad(): # Vô hiệu hóa tracking gradient[cite: 2]
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            # Lấy index của class có xác suất cao nhất
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")