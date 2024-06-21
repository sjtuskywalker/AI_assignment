import os
import torch
from PIL import Image
from torchvision import models
from torchvision import transforms
from classify import ViolenceClass
from pytorch_lightning.loggers import TensorBoardLogger

# 设置文件夹路径
folder_path = 'data/val'

# 初始化空列表
image_list = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为.jpg
    if filename.lower().endswith('.jpg'):
        # 将文件路径添加到列表中
        image_list.append(os.path.join(folder_path, filename))

# 此时，image_list 包含了所有.jpg文件的完整路径
violence_classifier = ViolenceClass()
transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

tensor_imgs = torch.randn(len(image_list), 3, 224, 224)

# 将所有图片转化为3*224*224向量
i = 0
for img_path in image_list:
    imgs = Image.open(img_path)
    imgs = transforms(imgs)
    tensor_imgs[i] = imgs
    i += 1

#得到长度为'n'的python列表（每个值为对应的预测类别，即整数0或1）
prediction = violence_classifier.classify(tensor_imgs)