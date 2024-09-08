import pickle
import os
import shutil
path='data\configure.pkl'
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
print(video_path_valid)
video_path_test=[]
for video_name in dataset['test']['id']:
    video_number = video_name.split('_')[1].split('$')[0]
    frame_number = video_name.split('_')[2].replace('$','')
    # 数据集路径列表
    video_path_test.append('.\data\Raw/video_{}/{}.mp4'.format(video_number, frame_number))


# 遍历所有视频文件路径

def change_path(file_paths,labels,dataset_type,k):
    
    # 目标文件目录
    target_directory = ".\\dataset\\video\\"+dataset_type
    s=['negative','neutral','positive']

    # 遍历文件路径和标签
    for path, label in zip(file_paths, labels):
        # 构建目标文件夹路径
        target_folder = os.path.join(target_directory, s[int(label)])
        # 构建目标文件路径
        target_path = os.path.join(target_folder, "video"+str(k)+".mp4")  # 替换为您想要的新文件名
    
        # 移动文件到目标路径并重新命名
        shutil.move(path, target_path)
        k=k+1
    return k
""" k=1369
k=change_path(video_path_valid,dataset['valid']['classification_labels_V'],'valid',k)
k=change_path(video_path_test,dataset['test']['classification_labels_V'],'test',k) """
        
    