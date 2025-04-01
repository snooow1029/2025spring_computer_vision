# ============================================================================
# File: model.py
# Date: 2025-03-11
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()

        # 卷積層 1: 輸入 3 通道 (RGB)，輸出 32 個 feature maps，kernel size 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批次標準化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化層 (2x2)

        # 卷積層 2: 輸入 32 通道，輸出 64 個 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 卷積層 3: 輸入 64 通道，輸出 128 個 feature maps
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 全連接層
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 壓平後進入 FC
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 輸出 10 個類別

        # dropout 層，防止過擬合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BN -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BN -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BN -> ReLU -> MaxPool

        x = torch.flatten(x, start_dim=1)  # 壓平成 1D
        x = F.relu(self.fc1(x))  # FC1 + ReLU
        x = self.dropout(x)  # Dropout
        x = F.relu(self.fc2(x))  # FC2 + ReLU
        x = self.fc3(x)  # 輸出層 (未加 softmax，因為 PyTorch 的 loss 會自動處理)
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        # load resnet18 model
        self.resnet = models.resnet18(pretrained=True)

        # modified the first conv layer to accept 3 channels (CIFAR-10)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # remove the maxpool layer (resnet18 has a maxpool layer after conv1)
        # this is to avoid downsampling the input too much
        self.resnet.maxpool = nn.Identity()

        # modify the last fully connected layer to output 10 classes (CIFAR-10)
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(self.resnet.fc.in_features),  # ✅ 新增 BN
            nn.ReLU(),
            nn.Dropout(0.5),  # ✅ Dropout 防止 overfitting
            nn.Linear(self.resnet.fc.in_features, 10)
        )


    def forward(self, x):
        return self.resnet(x)

    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
