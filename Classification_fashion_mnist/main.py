import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("use device: ", device)

# 读取npy数据
train_images = np.load('./data/train-images.npy')
train_labels = np.load('./data/train-labels.npy')
test_images = np.load('./data/t10k-images.npy')
test_labels = np.load('./data/t10k-labels.npy')

print("shape of train_images: ", train_images.shape)
print("shape of train_labels: ", train_labels.shape)
print("shape of test_images: ", test_images.shape)
print("shape of test_labels: ", test_labels.shape)

# 输入的tran_images, train_labels, test_images, test_labels已经为numpy格式
# 继承pytorch的Dataset，用于处理fashion-mnist数据集
class FashionMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx): # idx为索引, 该方法用于获取数据集中的数据和标签
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
# 定义数据集
# 用上面继承的FashionMNISTDataset类来定义数据集
train_dataset = FashionMNISTDataset(train_images, train_labels) # 传入numpy格式的数据
test_dataset = FashionMNISTDataset(test_images, test_labels)    # 传入numpy格式的数据

# 定义数据加载器
# 用torch.utils.data.DataLoader定义数据加载器
# 不用重新写DataLoader类，torch.utils.data.DataLoader已经定义好了，直接调用即可
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义网络模型, 用于MNIST分类
class CNN(nn.Module):
    # __init__方法是类的构造函数，用于初始化类的成员
    # forward方法定义了数据流向，即数据如何在网络层间传递
    # forward使用__init__中定义的网络层
    def __init__(self):
        super(CNN, self).__init__() # 调用父类的构造函数
        # 三层卷积层，卷积层使用批量归一化，两层池化层，两层全连接层
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),   # （bachsize, 1, 28, 28）->（bachsize, 16, 28, 28）
            # in_channels为输入的通道数, in_channels=1因为fashion-mnist的图片是灰度图，只有一个通道
            # out_channels为卷积核的数量, kernel_size为卷积核的大小, stride为步长, padding为填充, padding=(kernel_size-1)/2
            # stride=(1, 1)表示水平和竖直方向的步长都为1，padding=(1, 1)表示在水平和竖直方向都填充1个像素
            # padding=(input_size - kernel_size + 2*padding)/stride + 1, 这样卷积后的输出大小和输入大小相同(28*28)
            # 卷积核的大小一般为奇数，这样才能保证padding为整数
            nn.BatchNorm2d(num_features=16),    # (bachsize, 16, 28, 28) -> (bachsize, 16, 28, 28)
            nn.ReLU(),  # （bachsize, 16, 28, 28）->（bachsize, 16, 28, 28）
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))   # （bachsize, 16, 28, 28）->（bachsize, 16, 14, 14）
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # （bachsize, 16, 14, 14）->（bachsize, 32, 14, 14）
            nn.BatchNorm2d(num_features=32),    # （bachsize, 32, 14, 14）->（bachsize, 32, 14, 14）
            nn.ReLU(),  # （bachsize, 32, 14, 14）->（bachsize, 32, 14, 14）
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # （bachsize, 32, 14, 14）->（bachsize, 32, 7, 7）
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # （bachsize, 32, 7, 7）->（bachsize, 64, 7, 7）
            nn.BatchNorm2d(num_features=64),    # （bachsize, 64, 7, 7）->（bachsize, 64, 7, 7）
            nn.ReLU()   # （bachsize, 64, 7, 7）->（bachsize, 64, 7, 7）
        )

        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)  # （bachsize, 64*7*7）->（bachsize, 128）
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # （bachsize, 128）->（bachsize, 10）

    def forward(self, x):   # 定义数据流向，即数据如何在网络层间传递
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)   # 因为全连接层的输入是一维的，所以需要将卷积层的输出拉平
        x = self.fc1(x)
        x = self.fc2(x)
        return x