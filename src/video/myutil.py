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
import os 
import torch
import copy
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义collate_fn_rnn辅助函数
def collate_fn_rnn(batch):
	imgs_batch,label_batch=list(zip(*batch))
	imgs_batch=[imgs for imgs in imgs_batch if len(imgs)>0]
	label_batch=[torch.tensor(l) for l, imgs in zip(label_batch,imgs_batch) if len(imgs)>0]
	imgs_tensor=torch.stack(imgs_batch)
	labels_tensor=torch.stack(label_batch)
	return imgs_tensor,labels_tensor

def show(dataset,mean,std):
    #获取video_dataset_train中的一个数据
    imgs,label=dataset[0]
    if len(imgs)>0:
        print(imgs.shape,label,torch.min(imgs),torch.max(imgs))
    

    #显示一些帧数据
    import matplotlib.pylab as plt
    plt.figure(figsize=(10,10))
    for ii,img in enumerate(imgs[::5]):
        plt.subplot(3,1,ii+1)
        # 定义反归一化的转换操作
        denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])

        # 反归一化
        denormalized_img = denormalize(img)

        # 将张量转换为NumPy数组
        denormalized_img = denormalized_img.numpy()

        # 将数组维度重新排列为(H, W, C)
        denormalized_img = denormalized_img.transpose((1, 2, 0))
        
        plt.imshow(denormalized_img)
        plt.title(label)
    plt.show()
    
# 定义训练和验证函数，参数为模型、参数
def train_val(model, params):
	# 获取参数中的epochs、损失函数、优化器、训练数据集、验证数据集、检查是否符合、学习率调度器、权重路径
	num_epochs=params["num_epochs"]
	loss_func=params["loss_func"]
	opt=params["optimizer"]
	train_dl=params["train_dl"]
	val_dl=params["val_dl"]
	sanity_check=params["sanity_check"]
	lr_scheduler=params["lr_scheduler"]
	path2weights=params["path2weights"]

	# 初始化损失和指标历史记录
	loss_history={"train":[],"val":[]}
	metric_history={"train":[],"val":[]}
	# 初始化最佳模型权重
	best_model_wts=copy.deepcopy(model.state_dict())
	# 初始化最佳损失
	best_loss=float("inf")
	# 开始训练
	for epoch in range(num_epochs):
		# 获取当前学习率
		current_lr=get_lr(opt)
		print("Epoch {}/{}, current lr={}".format(epoch+1, num_epochs,current_lr))
		# 训练模式
		model.train()
		# 计算训练损失和指标
		train_loss,train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)
		# 记录训练损失和指标
		loss_history["train"].append(train_loss)
		metric_history["train"].append(train_metric)
		# 验证模式
		model.eval()
		# 不计算梯度
		with torch.no_grad():
			# 计算验证损失和指标
			val_loss,val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
		# 如果验证损失小于最佳损失，则更新最佳损失和最佳模型权重
		if val_loss<best_loss:
			best_loss=val_loss
			best_model_wts=copy.deepcopy(model.state_dict())
			# 保存最佳模型权重
			torch.save(model.state_dict(),path2weights)
			print("Copied best model weights")
		# 记录验证损失和指标
		loss_history["val"].append(val_loss)
		metric_history["val"].append(val_metric)
		# 根据验证损失调整学习率
		lr_scheduler.step(val_loss)
		# 如果当前学习率不等于获取到的学习率，则加载最佳模型权重
		if current_lr!=get_lr(opt):
			print("Loading best model weights")
			model.load_state_dict(best_model_wts)
		# 打印训练损失、验证损失和准确率
		print("Train loss:%.6f, dev loss:%.6f, accuracy:%.2f" % (train_loss, val_loss, 100*val_metric))
		print("-"*10)
	# 加载最佳模型权重
	model.load_state_dict(best_model_wts)
	# 返回模型、损失历史记录和指标历史记录
	return model, loss_history, metric_history


def get_lr(opt):
	for param_group in opt.param_groups:
		return param_group["lr"]

# 定义一个函数，计算输出和目标之间的指标，返回正确个数
def metrics_batch(output, target):
	# 计算输出中最大值的索引，并保持维度
	pred=output.argmax(dim=1,keepdim=True)
	# 计算正确个数
	corrects=pred.eq(target.view_as(pred)).sum().item()
	# 返回正确个数
	return corrects

# 定义损失函数，计算输出和目标之间的损失，并计算指标
def loss_batch(loss_func, output, target, opt=None):
	# 计算输出和目标之间的损失

 
	loss=loss_func(output, target)
	# 计算指标
	with torch.no_grad():
		metric_b=metrics_batch(output,target)
	# 如果有优化器，则更新参数
	if opt is not None:
		opt.zero_grad()
		loss.backward()
		opt.step()
	# 返回损失和指标
	return loss.item(), metric_b

# 定义损失函数，计算模型在数据集上的损失和指标
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False,opt=None):
	# 初始化损失和指标
	running_loss=0.0
	running_metric=0.0
	# 计算数据集长度
	len_data=len(dataset_dl.dataset)
	# 遍历数据集
	for xb,yb in tqdm(dataset_dl):
		# 将数据转换到设备
		xb=xb.to(device)
		yb=yb.to(device)

		# 计算模型输出
		output=model(xb)
		# 计算损失和指标
		loss_b,metric_b=loss_batch(loss_func,output,yb,opt)
		# 累加损失和指标
		running_loss+=loss_b
		if metric_b is not None:
			running_metric+=metric_b
		# 检查是否需要进行 sanity check
		if sanity_check is True:
			break
	# 计算平均损失和指标
	loss=running_loss/float(len_data)
	metric=running_metric/float(len_data)
	# 返回平均损失和指标
	return loss, metric

def plot_loss(loss_hist, metric_hist):
	num_epochs=len(loss_hist["train"])
	plt.title("Train-Val Loss")
	plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
	plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
	plt.ylabel("Loss")
	plt.xlabel("Training Epochs")
	plt.legend()
	plt.show()
	plt.title("Train-Val Accuracy")
	plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
	plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
	plt.ylabel("Accuracy")
	plt.xlabel("Training Epochs")
	plt.legend()
	plt.show()
 
