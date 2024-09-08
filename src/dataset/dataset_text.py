import torch
import pickle
""" from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.datasets.folder as folder
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
from PIL import Image
import myutil """

path='.\data\configure.pkl'
with open(path, 'rb') as f:
    dataset = pickle.load(f)
    
def write_text_and_labels_to_txt(file_path, text_list, label_list):
    with open(file_path, 'w', encoding='utf-8') as file:
        for text, label in zip(text_list, label_list):
            line = str(text) + '\t' + str(int(label)) + '\n'
            file.write(line)
string_list=dataset['train']['raw_text']    
print(max(len(string) for string in string_list))
string_list=dataset['valid']['raw_text']    
print(max(len(string) for string in string_list))
string_list=dataset['test']['raw_text']    
print(max(len(string) for string in string_list))
print(max(dataset['train']['raw_text'], key=len))
print(max(dataset['valid']['raw_text'], key=len))
print(max(dataset['test']['raw_text'], key=len))

""" write_text_and_labels_to_txt('./data/text_data/train.txt',dataset['train']['raw_text'],dataset['train']['classification_labels_T'])
write_text_and_labels_to_txt('./data/text_data/valid.txt',dataset['valid']['raw_text'],dataset['valid']['classification_labels_T'])
write_text_and_labels_to_txt('./data/text_data/test.txt',dataset['test']['raw_text'],dataset['test']['classification_labels_T'])   """


    
    
