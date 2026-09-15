import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from models import Generator, Discriminator, weights_init
from dataset import get_dataloader

# ==========================================
# 1. Thiết lập tham số (Hyperparameters)
# ==========================================
dataroot = "Practice_GAN/data/train" # Đường dẫn data 
batch_size = 128
image_size = 64
nz = 100        # Kích thước vector nhiễu (noise)
num_epochs = 50 # Số vòng lặp qua toàn bộ dataset
lr = 0.0002
beta1 = 0.5

# ==========================================
# Khối bảo vệ đa luồng (Tránh lỗi RuntimeError)
# ==========================================
if __name__ == '__main__':
    # Tạo thư mục lưu kết quả ảnh sinh ra
    os.makedirs("output_images", exist_ok=True)

    # Lựa chọn thiết bị chạy (Ưu tiên GPU nếu có, không thì CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    # ==========================================
    # 2. Khởi tạo Dữ liệu, Model, Loss & Optimizer
    # ==========================================
    # Load dataloader
    dataloader = get_dataloader(dataroot, batch_size, image_size)

    # Khởi tạo mô hình và áp dụng hàm chuẩn hóa trọng số
    netG = Generator(nz=nz).to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # Hàm Loss: Binary Cross Entropy
    criterion = nn.BCELoss()

    # Tạo vector nhiễu cố định để theo dõi độ tiến bộ của ảnh qua từng epoch
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Khai báo nhãn thật/giả
    real_label = 1.
    fake_label = 0.

    # Optimizer (Sử dụng Adam)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # ==========================================
    # 3. Vòng lặp huấn luyện (Training Loop)
    # ==========================================
    print("Bắt đầu quá trình huấn luyện...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            
            ##################################################
            # (A) Cập nhật mạng Discriminator
            # Mục tiêu: Tối đa hóa log(D(x)) + log(1 - D(G(z)))
            ##################################################
            netD.zero_grad()
            
            # --- 1. Train với ảnh thật ---
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # --- 2. Train với ảnh giả (do Generator sinh ra) ---
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Tính tổng lỗi của Discriminator và cập nhật trọng số
            errD = errD_real + errD_fake
            optimizerD.step()

            ##################################################
            # (B) Cập nhật mạng Generator
            # Mục tiêu: Tối đa hóa log(D(G(z)))
            ##################################################
            netG.zero_grad()
            # Đánh lừa Discriminator: Gán nhãn cho ảnh giả là "thật"
            label.fill_(real_label)  
            
            output = netD(fake)
            # Biến errG được tính toán ở đây
            errG = criterion(output, label) 
            errG.backward()
            D_G_z2 = output.mean().item()
            
            # Cập nhật trọng số của Generator
            optimizerG.step()

            # In log ra màn hình mỗi 50 batch để theo dõi
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

        # Sau mỗi epoch, tạo và lưu ảnh mẫu để xem quá trình học
        with torch.no_grad():
            fake_display = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake_display, f"output_images/fake_samples_epoch_{epoch}.png", normalize=True)

    print("Hoàn tất huấn luyện!")