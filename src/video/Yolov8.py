import cv2
from ultralytics import YOLO
import os
def find_most_frequent(lst):
    max_count = 0
    most_frequent = None

    for item in lst:
        count = lst.count(item)
        if count > max_count:
            max_count = count
            most_frequent = item

    return most_frequent


folder_path = 'dataset\\video\\valid'  # 文件夹路径

file_paths = []  # 存储文件路径的列表

def get_file_paths(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)  # 构建文件路径
            file_paths.append(file_path)

        for dir in dirs:
            dir_path = os.path.join(root, dir)  # 构建子文件夹路径
            get_file_paths(dir_path)  # 递归调用，遍历子文件夹

get_file_paths(folder_path)
model = YOLO('model\yolov8n.pt')
import csv



# CSV 文件路径
csv_file = 'src\\video\\result.csv'




# 打印所有文件路径
for file_path in file_paths:

    results = model(file_path, stream=True)  # generator of Results objects

    




