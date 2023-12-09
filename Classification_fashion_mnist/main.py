import numpy as np
import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_images = np.load('./data/train-images.npy')
train_labels = np.load('./data/train-labels.npy')
test_images = np.load('./data/t10k-images.npy')
test_labels = np.load('./data/t10k-labels.npy')

print("shape of train_images: ", train_images.shape)
