import torch
import pickle
from torch.utils.data import Dataset
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
import myutil


#视频
def set_V(path,model_type): 
    #定义自己的视频数据类
    class VideoDataset(Dataset):
        def __init__(self, data_root, labels, transform=None):
            self.data_root = data_root
            self.labels = labels
            self.transform = transform
            

        def __len__(self):
            return len(self.data_root)

        def __getitem__(self, idx):
            video_path = self.data_root[idx]
            frames = self.load_video_frames(video_path)
            
            #设置随机种子
            seed=np.random.randint(1e9)
            #定义一个空列表
            frames_tr=[]
            #遍历frames中的每一帧
            for frame in frames:
                #设置随机种子
                random.seed(seed)
                np.random.seed(seed)
                frame = Image.fromarray(frame)
                #对帧进行变换
                frame=self.transform(frame)
                #将变换后的帧添加到列表中
                frames_tr.append(frame)
            #如果列表不为空
            if len(frames_tr)>0:
                #将列表中的帧转换为tensor
                frames_tr=torch.stack(frames_tr)
            # 获取对应的标签
            label = int(self.labels[idx])
            
            return frames_tr, label
        #加载视频帧
        def load_video_frames(self, video_path):
            frames = []
            cap = cv2.VideoCapture(video_path)
            #循环读取帧
            while cap.isOpened():
                ret, frame = cap.read()
                #读取完毕，跳出循环
                if not ret:
                    break
                #将帧从BGR转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            #释放资源
            cap.release()
            
            #等距取15个图像
            step = int(len(frames) / 15)
            start = 0
            end = 15*step
            #索引
            indices = list(range(start, end, step))
            frames_selected = [frames [i] for i in indices]
            
            
            return frames_selected   
    #视频数据集路径
    #打开文件 
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    #train
    video_path_train=[]
    for video_name in dataset['train']['id']:
        video_number = video_name.split('_')[1].split('$')[0]
        frame_number = video_name.split('_')[2].replace('$','')
        # 数据集路径列表
        video_path_train.append('.\data\Raw/video_{}/{}.mp4'.format(video_number, frame_number))
    #vaild
    video_path_valid=[]
    for video_name in dataset['valid']['id']:
        video_number = video_name.split('_')[1].split('$')[0]
        frame_number = video_name.split('_')[2].replace('$','')
        # 数据集路径列表
        video_path_valid.append('.\data\Raw/video_{}/{}.mp4'.format(video_number, frame_number))
    #test
    video_path_test=[]
    for video_name in dataset['test']['id']:
        video_number = video_name.split('_')[1].split('$')[0]
        frame_number = video_name.split('_')[2].replace('$','')
        # 数据集路径列表
        video_path_test.append('.\data\Raw/video_{}/{}.mp4'.format(video_number, frame_number))
    
    #定义变换参数
    if model_type=="rnn":
        h,w=224,224
        mean=[0.485,0.456,0.406]
        std=[0.229,0.224,0.225]
    else:
        h,w=112,112
        mean=[0.43216,0.394666,0.37645]
        std=[0.22803,0.22145,0.216989]    




    #数据加强
    transform = transforms.Compose([
					transforms.Resize((h,w)),#大小
					transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
					transforms.ToTensor(),#张量
					transforms.Normalize(mean,std),#均值化
					])	

    # 使用自定义的数据集类创建数据集
    video_dataset_train = VideoDataset(video_path_train,dataset['train']['classification_labels_V'],transform=transform)#路径，标签，增强
    # train展示
    myutil.show(video_dataset_train,mean,std)
    
    video_dataset_valid = VideoDataset(video_path_valid,dataset['valid']['classification_labels_V'],transform=transform)#路径，标签，增强
    print(len(video_dataset_valid))
    #vaild展示
    myutil.show(video_dataset_valid,mean,std)
    
    video_dataset_test = VideoDataset(video_path_test,dataset['test']['classification_labels_V'],transform=transform)#路径，标签，增强
    print(len(video_dataset_test))
    #test展示
    myutil.show(video_dataset_test,mean,std)

    # 数据加载器
    #批次
    bacth=5
    video_loader_train = DataLoader(video_dataset_train, batch_size=bacth, shuffle=False,collate_fn=myutil.collate_fn_rnn)
    video_loader_valid = DataLoader(video_dataset_valid, batch_size=bacth, shuffle=False,collate_fn=myutil.collate_fn_rnn)
    video_loader_test = DataLoader(video_dataset_test, batch_size=bacth, shuffle=False,collate_fn=myutil.collate_fn_rnn)
    return video_loader_train, video_loader_valid, video_loader_test

#文本
def set_T(path):
    class TextDataset(Dataset):
        def __init__(self, text_data, labels=None):
            self.text_data = text_data
            self.labels = labels

        def __len__(self):
            return len(self.text_data)

        def __getitem__(self, index):
            text = self.text_data[index]
            label = self.labels[index] if self.labels is not None else None
            return text, label
    #打开文件 
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
        
    # 使用自定义的数据集类创建数据集
    txt_dataset_train = TextDataset(dataset['train']['raw_text'],dataset['train']['regression_labels_T'])
    print(len(txt_dataset_train))
    txt_dataset_valid = TextDataset(dataset['valid']['raw_text'],dataset['valid']['regression_labels_T'])
    print(len(txt_dataset_valid))
    txt_dataset_test = TextDataset(dataset['test']['raw_text'],dataset['test']['regression_labels_T'])
    print(len(txt_dataset_test))
    
    # 创建数据加载器
    #批次
    bacth=10
    txt_loader_train = DataLoader(txt_dataset_train, batch_size=bacth, shuffle=False)
    txt_loader_valid = DataLoader(txt_dataset_valid, batch_size=bacth, shuffle=False)
    txt_loader_test = DataLoader(txt_dataset_test, batch_size=bacth, shuffle=False)
    return txt_loader_train, txt_loader_valid, txt_loader_test

#音频
def set_A(path):
    class AudioDataset(Dataset):
        def __init__(self, audio_data, labels=None):
            self.audio_data = audio_data
            self.labels = labels

        def __len__(self):
            return len(self.audio_data)

        def __getitem__(self, index):
            audio = self.audio_data[index]
            label = self.labels[index] if self.labels is not None else None
            return audio, label
    #打开文件 
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    # 使用自定义的数据集类创建数据集
    audio_dataset_train = AudioDataset(dataset['train']['audio'],dataset['train']['regression_labels_A'])
    print(len(audio_dataset_train))
    audio_dataset_valid = AudioDataset(dataset['valid']['audio'],dataset['valid']['regression_labels_A'])
    print(len(audio_dataset_valid))
    audio_dataset_test = AudioDataset(dataset['test']['audio'],dataset['test']['regression_labels_A']) 
    print(len(audio_dataset_test))
    
    # 创建数据加载器
    #批次
    bacth = 10
    audio_loader_train = DataLoader(audio_dataset_train, batch_size=bacth, shuffle=False)
    audio_loader_valid = DataLoader(audio_dataset_valid, batch_size=bacth, shuffle=False) 
    audio_loader_test = DataLoader(audio_dataset_test, batch_size=bacth, shuffle=False)
    
    return audio_loader_train,audio_loader_valid,audio_loader_test  
    
    
    
    
    
    
    
    




 





