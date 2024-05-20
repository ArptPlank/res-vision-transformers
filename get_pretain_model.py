from transformers import ViTModel

import torch
# 初始化模型，这里以 `vit-base-patch16-224` 为例
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
# 指定权重文件的保存路径
save_path = './vit-base-patch16-224.pth'

# 保存模型权重
torch.save(model.state_dict(), save_path)
