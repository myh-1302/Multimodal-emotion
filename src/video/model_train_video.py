
from torch import optim
from torch.optim.lr_scheduler import  ReduceLROnPlateau
import myutil
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
from model_video import model_v
import dataset_construction as dc
model_name='rnn'
# 配置文件路径
path='.\data\configure.pkl'

#视频数据集
video_loader_train, video_loader_valid, video_loader_test=dc.set_V(path,model_name)
# 定义模型
model=model_v(model_name,3)
# 定义损失函数
loss_func=nn.CrossEntropyLoss(reduction="sum")
# 定义优化器
opt=optim.Adam(model.parameters(),lr=3e-5)
# 余弦退火学习率中LR的变化是周期性的，T_max是周期的1/2；eta_min(float)表示学习率的最小值，默认为0；
# last_epoch(int)代表上一个epoch数，该变量用来指示学习率是否需要调整。当last_epoch符合设定的间隔时，
# 就会对学习率进行调整。当为-1时，学习率设为初始值。
# lr_scheduler = CosineAnnealingLR(opt, T_max=20, verbose=True)
lr_scheduler=ReduceLROnPlateau(opt,mode="min",factor=0.5,patience=5,verbose=1)

#2. 调用myutils中的train_val辅助函数训练模型
params_train={
	"num_epochs":20,
	"optimizer":opt,
	"loss_func":loss_func,
	"train_dl":video_loader_train,
	"val_dl":video_loader_valid,
	"sanity_check":True,
	"lr_scheduler":lr_scheduler,
	"path2weights":"./model/video_"+model_name+".pt",
}
model,loss_hist,metric_hist=myutil.train_val(model,params_train)
#运行完前面的代码片段后，训练将开始，您应该会在屏幕上看到它的进度。
#训练结束后，绘制训练进度
myutil.plot_loss(loss_hist, metric_hist)
# 前面的片段将显示一个损失和准确性的图。
