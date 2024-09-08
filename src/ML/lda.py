import numpy as np
from sklearn.manifold import TSNE
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# 读取.pkl文件
with open('data\configure.pkl', 'rb') as file:
    data = pickle.load(file)
data_text = data['train']['text']
print(data_text.shape)
data_audio = data['train']['audio']
print(data_audio.shape)
data_vision = data['train']['vision']
print(data_vision.shape)


data_text_2d = data_text.reshape(data_text.shape[0], -1)
data_audio_2d = data_audio.reshape(data_audio.shape[0], -1)
data_vision_2d = data_vision.reshape(data_vision.shape[0], -1)
# 创建LDA模型
lda = LatentDirichletAllocation(n_components=2)

# 使用LDA对文本数据进行降维
# Preprocess the data by removing negative values
data_text_2d[data_text_2d < 0] = 0
data_audio_2d[data_audio_2d < 0] = 0
data_vision_2d[data_vision_2d < 0] = 0

data_text_lda = lda.fit_transform(data_text_2d)

# 使用LDA对音频数据进行降维
data_audio_lda = lda.fit_transform(data_audio_2d)

# 使用LDA对视觉数据进行降维
data_vision_lda = lda.fit_transform(data_vision_2d)

# 打印降维后的数据形状
print(data_text_lda.shape)
print(data_audio_lda.shape)
print(data_vision_lda.shape)

# 绘制降维后的数据
plt.scatter(data_text_lda[:, 0], data_text_lda[:, 1], label='Text')
plt.scatter(data_audio_lda[:, 0], data_audio_lda[:, 1], label='Audio')
plt.scatter(data_vision_lda[:, 0], data_vision_lda[:, 1], label='Vision')
plt.legend()
plt.show()
