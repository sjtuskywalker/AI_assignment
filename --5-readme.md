## 接口调用说明

接口调用使用的是call_example.py，它从data/val中读入图片，转化成python tensor，然后将其作为输入，由classify.py来print最终结果，并将预测结果返回。

调用时的五个主要步骤：

（1）from classify import ViolenceClass 

（2）设置文件路径（此处默认data/val）

（3）将图片读入并转化为向量tensor_imgs

（4）得到ViolenceClass实例

（5）将tensor_imgs作为ViolenceClass的classify方法的输入，并得到结果prediction



以下是代码

```python
import os
import torch
from PIL import Image
from torchvision import models
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from classify import ViolenceClass  #从classify中import测试需要的接口类

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
```



