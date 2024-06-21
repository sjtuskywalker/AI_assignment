import torch
from torch import nn
import numpy as np
from torchvision import models
from torchvision import transforms
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule

class ViolenceClass:
    def __init__(self):
        # 加载预训练模型
        ckpt_path = "resnet18_pretrain_test-epoch=02-val_loss=0.03.ckpt"
        self.model = ViolenceClassifier.load_from_checkpoint(ckpt_path).to('cuda:0')
    
    def classify(self, tensor_imgs):
        #设置处于评估模式
        self.model.eval()
        
        with torch.no_grad():
            tensor_imgs = tensor_imgs.to('cuda:0')
            outputs = self.model(tensor_imgs)
            _, predicted = torch.max(outputs, 1)
            print(predicted.tolist())

        return predicted.tolist()

#ViolenceClass类中需要使用ViolenceClassifier的load_from_checkpoint方法故此处定义ViolenceClassifier类

class ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 定义优化器
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log('test_acc', acc)
        return acc