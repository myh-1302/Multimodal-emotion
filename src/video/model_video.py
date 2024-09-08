import dataset_construction as dc
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
from torch import nn
from torchvision import models
# 为视频分类定义两个模型
def model_v(model_type,num_classes):
 
    # 定义Resnet50Rnn类，继承nn.Module
    class Resnet50Rnn(nn.Module):
        # 初始化函数，参数为params_model
        def __init__(self,params_model):
            super(Resnet50Rnn,self).__init__()
            # 获取参数model中的参数
            num_classes=params_model["num_classes"]
            dr_rate=params_model["dr_rate"]
            pretrained=params_model["pretrained"]
            rnn_hidden_size=params_model["rnn_hidden_size"]
            rnn_num_layers=params_model["rnn_num_layers"]
            # 加载resnet18模型，并将其设置为baseModel
            baseModel=models.resnet50(pretrained=pretrained)
            # 获取baseModel的输入特征数
            num_features=baseModel.fc.in_features
            # 将baseModel的fc设置为Identity
            baseModel.fc=Identity()
            # 将baseModel设置为self.baseModel
            self.baseModel=baseModel
            # 将dropout设置为nn.Dropout，参数为dr_rate
            self.dropout=nn.Dropout(dr_rate)
            # 将rnn设置为nn.LSTM，参数为num_features,rnn_hidden_size,rnn_num_layers
            self.rnn=nn.LSTM(num_features,rnn_hidden_size,rnn_num_layers)
            # 将fc1设置为nn.Linear，参数为rnn_hidden_size, num_classes
            self.fc1=nn.Linear(rnn_hidden_size, num_classes)
        # 定义前向传播函数，参数为x
        def forward(self,x):
            # 获取x的形状
            b_z,ts,c,h,w=x.shape
            # 初始化ii为0
            ii=0
            # 将baseModel的输入设置为x[:,ii]
            y=self.baseModel((x[:,ii]))
            
            # 将y设置为rnn的输入，并获取输出和隐藏层状态
            output,(hn,cn)=self.rnn(y.unsqueeze(1))
            
            # 遍历ts次
            for ii in range(1,ts):
                # 将baseModel的输入设置为x[:,ii]
                y=self.baseModel((x[:,ii]))
                # 将y设置为rnn的输入，并获取输出和隐藏层状态
                out,(hn,cn)=self.rnn(y.unsqueeze(1),(hn,cn))
            
            # 将输出设置为dropout的输入，并获取输出
            out=self.dropout(out[:, -1])
            
            # 将输出设置为fc1的输入，并获取输出
            out=self.fc1(out)
            
            # 返回输出
            return out
    # 定义Identity类，继承nn.Module
    class Identity(nn.Module):
        # 初始化函数
        def __init__(self):
            super(Identity,self).__init__()
        # 定义前向传播函数，参数为x
        def forward(self,x):
            # 返回x
            return x
    #使用条件语句来定义任意一个模型
    if model_type=="rnn":
        params_model={
            "num_classes":num_classes,
            "dr_rate":0.1,
            "pretrained":True,
            "rnn_num_layers":1,
            "rnn_hidden_size":100,
            }
        model=Resnet50Rnn(params_model)
    else:
        # 加载预训练模型
        model=models.video.r3d_18(pretrained=True,progress=False)
        # 获取模型的特征维度
        num_features=model.fc.in_features
        # 将模型的最后一层替换为线性层，并设置输出维度为num_classes
        model.fc=nn.Linear(num_features,num_classes)
    #将模型移到GPU设备
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    return model







