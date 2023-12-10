import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import CNN   # 导入model.py中定义的CNN类
import os

# 定义超参数
num_epochs = 10
num_classes = 10
batch_size = 64
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("use device: ", device)

# 读取npy数据
train_images = np.load('./data/train-images.npy')
train_labels = np.load('./data/train-labels.npy')
test_images = np.load('./data/t10k-images.npy')
test_labels = np.load('./data/t10k-labels.npy')

# print("shape of train_images: ", train_images.shape)
# print("shape of train_labels: ", train_labels.shape)
# print("shape of test_images: ", test_images.shape)
# print("shape of test_labels: ", test_labels.shape)

# 输入的tran_images, train_labels, test_images, test_labels已经为numpy格式
# 继承pytorch的Dataset，用于处理fashion-mnist数据集
class FashionMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # 定义的CNN模型的输入为(batch_size, 1, 28, 28)
        # images的shape为torch.Size([bachsize, 784]), labels的shape为torch.Size([bachsize])
        # 需要将images的shape转换为torch.Size([bachsize, 1, 28, 28])
        self.images = images.reshape(-1, 1, 28, 28) # -1表示该维度由其他维度推断得到
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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义网络模型
model = CNN().to(device)    # 将模型加载到device中，即加载到GPU或CPU中

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()   # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    # Adam优化器

# 训练模型
def train():
    total_step = len(train_loader)  # 计算总共有多少个batch
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader): # 用enumerate()函数将train_loader转换成索引-数据对
            images = images.float().to(device)  # 将数据加载到device中
            labels = labels.to(device)  # 将数据加载到device中

            print("images.shape: ", images.shape)
            print("labels.shape: ", labels.shape)

            # 前向传播
            outputs = model(images) # outputs的shape为(batch_size, 10)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()   # 将梯度归零
            loss.backward() # 反向传播计算梯度
            optimizer.step()    # 更新参数

            if (i+1) % 100 == 0:    # 每100个batch打印一次日志
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    # 保存模型
    if not os.path.exists('./model'):  # 如果./model文件夹不存在，则创建
        os.makedirs('./model')
    torch.save(model.state_dict(), './model/model.ckpt')   # 保存模型参数

# 测试模型
def test():
    model.load_state_dict(torch.load('./model/model.ckpt'))   # 加载模型参数
    model.eval()    # 将模型设置为评估模式，即计算过程中不要dropout和batch normalization
    with torch.no_grad():   # 不计算梯度
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)  # 将数据加载到device中
            labels = labels.to(device)  # 将数据加载到device中
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1) # 取得分最高的那个类
            total += labels.size(0) # labels.size(0)为batch_size
            correct += (predicted == labels).sum().item()   # 记录正确的个数
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    train()
    test()