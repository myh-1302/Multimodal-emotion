import pickle
import os
import shutil
import re
from moviepy.editor import VideoFileClip


path='data\configure.pkl'
#打开文件 
with open(path, 'rb') as f:
    dataset = pickle.load(f)
def get_audio(dataset_type,labels,num):
    #原始文件目录
    input_directory = ".\\dataset\\video\\"+dataset_type
    # 目标文件目录
    target_directory = ".\\dataset\\audio\\"+dataset_type
    
    s=['negative','neutral','positive']
    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 提取数字
            numbers = re.findall(r'\d+', file_path)
            # 提取特定部分的数字
            if numbers:
                target_number = numbers[0]
            
            target_folder = os.path.join(target_directory, s[int(labels[int(target_number)-num])])
            # 如果目标文件夹不存在，则创建
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            # 构建目标文件路径
            audio_path  = os.path.join(target_folder, "audio"+str(int(target_number))+".wav")  # 替换为您想要的新文件名
            # 创建VideoFileClip对象
            video_clip = VideoFileClip(file_path)

            # 提取音频并保存为文件
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_path)
#get_audio('train',dataset['train']['classification_labels_A'])
#print(dataset['valid']['classification_labels_A'])
#print(len(dataset['test']['classification_labels_A']))
#get_audio('valid',dataset['valid']['classification_labels_A'],1369)
get_audio('test',dataset['test']['classification_labels_A'],1825)
#get_audio('valid',dataset['valid']['classification_labels_A'])


            
""" from collections import defaultdict

# 列表
my_list = dataset['train']['classification_labels_A']

# 创建字典来存储相同元素的序号
index_dict = defaultdict(list)

# 遍历列表并记录元素的序号
for index, element in enumerate(my_list):
    index_dict[element].append(index)

input_directory=".\\dataset\\audio\\train"
s=['negative','neutral','positive']
# 打印相同元素的序号
for element, indices in index_dict.items():
    print(element, indices)
    path=os.path.join(input_directory,s[int(element)])
    for root, dirs, files in os.walk(path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 提取数字
            numbers = re.findall(r'\d+', file_path)
            # 提取特定部分的数字
            if numbers:
                target_number = numbers[0] """
            
    


    