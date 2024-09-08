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
from utils import read_data, kmeans_plot
import numpy as np
from sklearn.manifold import TSNE
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


data_text_2d,data_audio_2d,data_vision_2d,data_text_label,data_audio_label,data_vision_label = read_data('train')


# 聚类文本数据
kmeans_text = KMeans(n_clusters=3)
text_clusters = kmeans_text.fit_predict(data_text_2d)

# 聚类音频数据
kmeans_audio = KMeans(n_clusters=3)
audio_clusters = kmeans_audio.fit_predict(data_audio_2d)

# 聚类视觉数据
kmeans_vision = KMeans(n_clusters=3)
vision_clusters = kmeans_vision.fit_predict(data_vision_2d)


# 降维
tsne = TSNE(n_components=2)
text_2d = tsne.fit_transform(data_text_2d)
audio_2d = tsne.fit_transform(data_audio_2d)
vision_2d = tsne.fit_transform(data_vision_2d)

kmeans_plot(text_2d,'Text',text_clusters)
kmeans_plot(audio_2d,'Audio',audio_clusters)
kmeans_plot(vision_2d,'Vision',vision_clusters)



