import numpy as np
from sklearn.manifold import TSNE
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def read_data(type):
    # 读取.pkl文件
    with open('data\configure.pkl', 'rb') as file:
        data = pickle.load(file)
    data_text = data[type]['text']
    print(data_text.shape)
    data_audio = data[type]['audio']
    print(data_audio.shape)
    data_vision = data[type]['vision']
    print(data_vision.shape)
    # 将数据重塑为2D
    data_text_2d = data_text.reshape(data_text.shape[0], -1)
    data_audio_2d = data_audio.reshape(data_audio.shape[0], -1)
    data_vision_2d = data_vision.reshape(data_vision.shape[0], -1)
    # 标签
    data_text_label = data[type]['classification_labels_T']
    print(data_text_label.shape)
    data_audio_label = data[type]['classification_labels_A']
    print(data_audio_label.shape)
    data_vision_label = data[type]['classification_labels_V']
    print(data_vision_label.shape)
    return data_text_2d,data_audio_2d,data_vision_2d,data_text_label,data_audio_label,data_vision_label

def kmeans_plot(text_2d,Text,text_clusters):
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    # 绘制降维后的数据图像
    # 设置图片尺寸和分辨率
    plt.figure(figsize=(6,4), dpi=300) 
    # Plotting text clustering
    plt.scatter(text_2d[:, 0], text_2d[:, 1], c=text_clusters)
    plt.title(Text+ 'Clustering')
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

    plt.savefig('output/ML/'+Text+'.png', format='png', bbox_inches='tight')  # Save as PNG format
    plt.show()
    