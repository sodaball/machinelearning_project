# Report3 - 服装分类

FashionMNIST 是一个替代 [MNIST 手写数字集](https://link.zhihu.com/?target=http%3A//yann.lecun.com/exdb/mnist/)的图像数据集。

数据来源：[fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

 它是由 Zalando（一家德国的时尚科技公司）旗下的研究部门提供。其涵盖了来自 10 种类别的共 7 万个不同商品的正面图片。

FashionMNIST 的大小、格式和训练集/测试集划分与原始的 MNIST 完全一致。60000/10000 的训练测试数据划分，28x28 的灰度图片。你可以直接用它来测试你的机器学习和深度学习算法性能，**且不需要改动任何的代码**。

这个数据集的样子大致如下（每个类别占三行）：

![fashion-mnist.jpg](images/fashion-mnist.jpg)

以下是Fashion-MNIST数据集中的类别标签：

1. T-shirt/top（T恤/上衣）
2. Trouser（裤子）
3. Pullover（套头衫）
4. Dress（裙子）
5. Coat（外套）
6. Sandal（凉鞋）
7. Shirt（衬衫）
8. Sneaker（运动鞋）
9. Bag（包）
10. Ankle boot（踝靴）




## 步骤：

### 错误记录

错误一：

模型的权重是`FloatTensor`类型，但输入数据是`LongTensor`类型

```python
# 训练模型
def train():
    total_step = len(train_loader)  # 计算总共有多少个batch
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader): # 用enumerate()函数将train_loader转换成索引-数据对
            images = images.to(device)  # 将数据加载到device中
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
```

在输入image时加上`.float()`，修正后的代码：

```python
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
```



