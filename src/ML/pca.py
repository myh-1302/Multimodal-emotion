import numpy as np
from sklearn.manifold import TSNE
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import read_data
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
data_text_2d,data_audio_2d,data_vision_2d,data_text_label,data_audio_label,data_vision_label = read_data('test')
import io
with open('x_feats_concatenated.txt', 'r') as file:
    data = file.read()
with open('column_indices.txt', 'r') as file:
    data1 = file.read()   
# 将字符串数据转换为NumPy数组
data_array = np.genfromtxt(io.StringIO(data1), delimiter=' ')

# 将浮点数转换为整数
data_array = data_array.astype(int)
data = np.genfromtxt(io.StringIO(data), delimiter=' ')
pca = PCA(n_components=2) 
embedded_data = pca.fit_transform(data)
# 绘制带有类别的散点图
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=data_array)
plt.title('Embedded Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
# 对文本特征进行PCA降维
pca_text = PCA(n_components=2)  # 指定要保留的主成分个数
data_text_reduced = pca_text.fit_transform(data_text_2d)

# 对音频特征进行PCA降维
pca_audio = PCA(n_components=2)  # 指定要保留的主成分个数
data_audio_reduced = pca_audio.fit_transform(data_audio_2d)

# 对视觉特征进行PCA降维
pca_vision = PCA(n_components=2)  # 指定要保留的主成分个数
data_vision_reduced = pca_vision.fit_transform(data_vision_2d)

# 打印降维后数据的形状
print(data_text_reduced.shape)
print(data_audio_reduced.shape)
print(data_vision_reduced.shape)
# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
# 绘制降维后的数据图像
# 设置图片尺寸和分辨率
plt.figure(figsize=(6,4), dpi=300) 
plt.scatter(data_text_reduced[:, 0], data_text_reduced[:, 1], label='Text', color='blue', alpha=0.7)
plt.scatter(data_audio_reduced[:, 0], data_audio_reduced[:, 1], label='Audio', color='green', alpha=0.7)
plt.scatter(data_vision_reduced[:, 0], data_vision_reduced[:, 1], label='Vision', color='red', alpha=0.7)
# 美化图例
plt.legend(fontsize='large', loc='upper right', frameon=False)
plt.title('t-SNE Visualization', weight='bold')
plt.xlabel('Dimension 1', weight='bold')
plt.ylabel('Dimension 2', weight='bold')


# 加粗坐标轴
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)  # 底部边界线粗细
ax.spines['left'].set_linewidth(2)  # 左边界线粗细
ax.spines['right'].set_linewidth(2)  # 右边界线粗细
ax.spines['top'].set_linewidth(2)  # 顶部边界线粗细
# 设置小刻度
ax = plt.gca()
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# 显示小刻度
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=6)

plt.savefig('output/ML/pca.png', format='png', bbox_inches='tight')  # Save as PNG format

plt.show()

