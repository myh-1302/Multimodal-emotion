from MSA_FET import FeatureExtractionTool
from MSA_FET import run_dataset
from MMSA import MMSA_test
from MMSA import MMSA_run, get_config_regression
import pickle
import numpy as np
import torch
import csv
import pandas as pd
import os
""" path='yanzi.mp4'
 # 使用一个自定义的配置文件 "custom_config.json" 
fet = FeatureExtractionTool("src\config.json",log_dir='log\mutli',tmp_dir='tmp\\tmp')
#提取文件
feature1 = fet.run_single(in_file=path,out_file="feature.pkl",text="好不好？燕子，你要开心，你要幸福，好不好？开心啊，幸福。你的世界以后没有我了，没关系，你要自己幸福。") """  
""" # 加载特征文件 
with open('data\configure.pkl', 'rb') as f:
    features = pickle.load(f)
print(features['train']['audio'][1].shape)
print(features['train']['text'][1].shape)
print(features['train']['vision'][1].shape)
print(features['train']['text_bert'][1].shape)
path="yanzi.mp4"

with open('feature.pkl', 'rb') as f:
    features = pickle.load(f)

# 使用特征
# 假设你的特征文件有一个名为 'feature1' 的键
print(features['audio'].shape)
print(features['text'].shape)
print(features['vision'].shape) 
print(features['text_bert'].shape) """

config = get_config_regression('mlmf', 'sims')

config['train_mode']='classification'
config['batch_size']=16
config['num_classes']=3
print(config)
output = MMSA_test(config=config, weights_path='model\mlmf-sims.pth', feature_path='feature.pkl', gpu_id=-1)










