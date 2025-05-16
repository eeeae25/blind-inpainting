import torch
import torch.nn as nn
import torchvision.models as models

# -------------------- Attention Modules -------------------- #
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = torch.mean(x, dim=(2, 3)).view(b, c)
        max_pool, _ = torch.max(x.view(b, c, -1), dim=2)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        out = self.relu(self.fc1(concat))
        out = self.sigmoid(self.fc2(out)).view(b, c, 1, 1)
        return x * out

# -------------------- Feature Extraction Branch -------------------- #
class FeatureBranch(nn.Module):
    def __init__(self, backbone):
        super(FeatureBranch, self).__init__()
        self.backbone = backbone
        self.spatial_attn = SpatialAttention()
        self.channel_attn = None  # will be defined later based on output channels

    def forward(self, x):
        x = self.backbone(x)
        if self.channel_attn is None:
            self.channel_attn = ChannelAttention(x.size(1)).to(x.device)
        x = self.spatial_attn(x)
        x = self.channel_attn(x)
        return x  # 保持空间维度用于后续 mask 输出

# -------------------- Full QualityNet Model (Modified for Mask Output) -------------------- #
class QualityNet(nn.Module):
    def __init__(self):
        super(QualityNet, self).__init__()
        # Use pretrained ResNet50 and EfficientNet-B0
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        self.branch1 = FeatureBranch(nn.Sequential(*list(resnet.children())[:-2]))
        self.branch2 = FeatureBranch(nn.Sequential(*list(efficientnet.features)))

        # 融合后上采样模块生成热力图
        self.mask_head = nn.Sequential(
            nn.Conv2d(2048 + 1280, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()  # 输出值范围在0~1之间的概率图
        )

    def forward(self, x):
        f1 = self.branch1(x)  # shape: [B, 2048, H/32, W/32]
        f2 = self.branch2(x)  # shape: [B, 1280, H/32, W/32]
        fused = torch.cat([f1, f2], dim=1)  # [B, 3328, H/32, W/32]
        heatmap = self.mask_head(fused)  # [B, 1, H/32, W/32]
        heatmap = nn.functional.interpolate(heatmap, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        return heatmap  # 返回像素级热力图

# -------------------- Test -------------------- #
if __name__ == "__main__":
    model = QualityNet()
    dummy_input = torch.randn(2, 3, 224, 224)  # batch of 2 RGB images
    output = model(dummy_input)
    print("Predicted heatmap shape:", output.shape)  # Expected: [2, 1, 224, 224]
