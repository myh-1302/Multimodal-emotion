import numpy as np
from sklearn.manifold import TSNE
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from utils import read_data
from sklearn import svm
from tqdm import tqdm
from sklearn.model_selection import KFold

train_text, train_audio, train_vision, train_text_label, train_audio_label, train_vision_label = read_data('train')
val_text, val_audio, val_vision, val_text_label, val_audio_label, val_vision_label = read_data('valid')
test_text, test_audio, test_vision, test_text_label, test_audio_label, test_vision_label = read_data('test')

# 创建一个SVM分类器
classifier = svm.SVC()

# 使用k折交叉划分train
kfold = KFold(n_splits=5)
train_indices = np.arange(len(train_text))

for train_index, _ in tqdm(kfold.split(train_text)):
    train_text_fold = train_text[train_index]
    train_text_label_fold = train_text_label[train_index]
    classifier.fit(train_text_fold, train_text_label_fold)
    
    

# 使用SVM分类器进行预测
predictions = classifier.predict(test_text)

# 评估性能
accuracy = np.mean(predictions == test_text_label)
print("准确率:", accuracy)




