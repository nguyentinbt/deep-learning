import torch
from torch import nn
from torchvision.models import resnet50

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        # 1. Trích xuất đặc trưng với ResNet50
        self.backbone = resnet50()
        del self.backbone.fc # Bỏ lớp Fully Connected cuối của ResNet
        
        # Lớp Conv để giảm chiều của feature map từ 2048 xuống hidden_dim (256)
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        
        # 2. Kiến trúc Transformer cốt lõi
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        
        # 3. Prediction Heads (Phân loại và dự đoán Bounding box)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1) # +1 cho lớp "không có gì" (background)
        self.linear_bbox = nn.Linear(hidden_dim, 4) # 4 tọa độ: [x, y, w, h]
        
        # 4. Object Queries & Positional Encodings
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim)) # Giới hạn max 100 đối tượng
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # inputs shape: [batch_size, 3, H, W]
        
        # Đưa ảnh qua Backbone (ResNet)
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Giảm chiều xuống hidden_dim (256)
        h = self.conv(x)
        
        # Tạo Positional Encoding (Mã hóa vị trí 2D)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        # Đưa vào Transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # Trả về kết quả dự đoán
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}