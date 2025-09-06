from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from mask_predict_model import QualityNet

# -------------------- CelebA 图像预处理 -------------------- #
# 假设 CelebA 图像目录
celeba_dir = "/path/to/celeba/img_align_celeba/"
img_file = "000001.jpg"  # 示例图片
img_path = os.path.join(celeba_dir, img_file)

# 加载并预处理图像
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 转为Tensor，并自动归一化到[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准
                         std=[0.229, 0.224, 0.225])
])

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度 → [1, 3, 224, 224]

# -------------------- 模型推理并显示热力图 -------------------- #
model = QualityNet()
model.eval()
with torch.no_grad():
    heatmap = model(input_tensor)

# 显示热力图
heatmap_np = heatmap.squeeze().cpu().numpy()
plt.imshow(heatmap_np, cmap="jet")
plt.colorbar()
plt.title("Predicted Damage Heatmap (CelebA)")
plt.show()
