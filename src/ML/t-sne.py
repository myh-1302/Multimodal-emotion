import numpy as np
from sklearn.manifold import TSNE
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from utils import read_data
import io
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
with open('x_feats_concatenated.txt', 'r') as file:
    data = file.read()
with open('column_indices.txt', 'r') as file:
    data1 = file.read()   
# 将字符串数据转换为NumPy数组
data_array = np.genfromtxt(io.StringIO(data1), delimiter=' ')

# 将浮点数转换为整数
data_array = data_array.astype(int)

print(data_array)
# Convert data to numpy array
data = np.genfromtxt(io.StringIO(data), delimiter=' ')
tsne = TSNE(n_components=2, random_state=42)
embedded_data = tsne.fit_transform(data)
# 绘制带有类别的散点图
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=data_array)
plt.title('Embedded Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()


# 应用 t-SNE









data_text_2d,data_audio_2d,data_vision_2d,data_text_label,data_audio_label,data_vision_label = read_data('test')

# 应用 t-SNE
tsne_text = TSNE(n_components=2, random_state=42)
tsne_audio = TSNE(n_components=2, random_state=42)
tsne_vision = TSNE(n_components=2, random_state=42)

embedded_text = tsne_text.fit_transform(data_text_2d)
embedded_audio = tsne_audio.fit_transform(data_audio_2d)
embedded_vision = tsne_vision.fit_transform(data_vision_2d)
plt.scatter(embedded_audio[:, 0], embedded_audio[:, 1], c=data_audio_label)
plt.title('Embedded Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show() 
""" # 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
# 绘制降维后的数据图像
# 设置图片尺寸和分辨率
plt.figure(figsize=(6,4), dpi=300) 
plt.scatter(embedded_text[:, 0], embedded_text[:, 1], label='Text', color='blue', alpha=0.7)
plt.scatter(embedded_audio[:, 0], embedded_audio[:, 1], label='Audio', color='green', alpha=0.7)
plt.scatter(embedded_vision[:, 0], embedded_vision[:, 1], label='Vision', color='red', alpha=0.7)
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

plt.savefig('output/ML/t_sne.png', format='png', bbox_inches='tight')  # Save as PNG format

plt.show()  """

""" import matplotlib.pyplot as plt

# Read CSV file
data1 = pd.read_csv('ft_mf.csv')
data2 = pd.read_csv('fa_mf.csv')
data3 = pd.read_csv('fv_mf.csv')

# Extract features and labels
features1 = data1.iloc[:, :].values
features2 = data2.iloc[:, :].values
features3 = data3.iloc[:, :].values


# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embedded_data1 = tsne.fit_transform(features1)
embedded_data2 = tsne.fit_transform(features2)
embedded_data3 = tsne.fit_transform(features3)

# Plot scatter plot
plt.scatter(embedded_data1[:, 0], embedded_data1[:, 1], label='Text', color='blue', alpha=0.7)
plt.scatter(embedded_data2[:, 0], embedded_data2[:, 1], label='Audio', color='green', alpha=0.7)
plt.scatter(embedded_data3[:, 0], embedded_data3[:, 1], label='Vision', color='red', alpha=0.7)
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

plt.savefig('output/ML/mlmf.png', format='png', bbox_inches='tight')  # Save as PNG format

plt.show() """


