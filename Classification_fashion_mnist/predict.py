import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import CNN_model  # 请确保 model.py 中有 CNN_model 类的定义

# 定义处理单张图片的函数
def process_image(image_path):
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    image = image.resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 PyTorch 的 Tensor
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 添加一个维度，模拟批次大小为1, shape为(1, 1, 28, 28)
    return image

# 加载模型
model = CNN_model()
model.load_state_dict(torch.load('./model/model.ckpt'))
model.eval()

# 读取一张图片并进行预测
image_path = './web_scraper/img_save/5.jpg'  # 替换为图片路径
image = process_image(image_path)
with torch.no_grad():
    output = model(image.float())
    probabilities = F.softmax(output, dim=1)
    _, predicted_class = torch.max(output, 1)

'''
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
'''
total_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 打印预测结果, predicted_class是一个Tensor，需要使用predicted_class.item()来获取数值
# 并将数值转为对应的英文class
# 打印每个类别的概率值
for i, prob in enumerate(probabilities.squeeze().numpy()):
    print(f'Class {total_classes[i]}: {prob * 100:.2f}%')

print('Predicted: ', total_classes[predicted_class.item()])
print('Probabilities: ', probabilities)


