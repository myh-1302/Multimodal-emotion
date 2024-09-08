import glob
import re
import numpy as np
from ultralytics import YOLO
import csv
import pandas as pd

def write_to_csv(preds):
    # 读取已有的CSV文件
    df = pd.read_csv('predictions.csv')

    # 添加新的列
    df['yolov8'] = preds

    # 写入CSV文件
    df.to_csv('predictions.csv', index=False)

def write_to_txt(results):
    # 打开文件以写入结果
    with open('results.txt', 'w') as file:
        # 将每个结果写入文件的新行
        for result in results:
            file.write(str(result) + '\n')

# 加载预训练的 YOLOv8n 模型
#model = YOLO('runs\\classify\\train3\\weights\\best.pt')

folder_path = "dataset\\image\\test\\negative"
jpg_files = glob.glob(folder_path + "/*.jpg")
folder_path = "dataset\\image\\test\\positive"
jpg_files += glob.glob(folder_path + "/*.jpg")
folder_path = "dataset\\image\\test\\neutral"
jpg_files += glob.glob(folder_path + "/*.jpg")

def extract_number(file_path):
    numbers = re.findall(r'\d+', file_path)
    if numbers:
        extracted_number = int(numbers[0])
        return (extracted_number + 14) // 15
    else:
        return None

d=[]
for file in jpg_files:
    d.append(extract_number(file))
sorted_indices = np.argsort(d)
# 对d进行排序
d_sorted = sorted(d)
# 使用sorted_indices对jpg_files进行排序
jpg_files = np.array(jpg_files)[sorted_indices]
results = []
""" for file,i in zip(jpg_files,d_sorted):
# 对来源进行推理
    result = model(file)  # Results 对象列表
    for r in result:
        results.append(r.numpy().probs.top1) """

# 读取txt文件
with open('results.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        result = line.strip()
        results.append(int(result))
from collections import Counter
g=[]
# 使用collections.Counter统计d_sorted中不同元素的数量
counter = Counter(d_sorted)
a=0
b=-1

for i in range(1825,2282):
    b += counter[i]
    print(b)
    d=results[a:b]
    print(d)
    counter1 = Counter(d)
    print(counter1)

    # 使用most_common方法找出出现次数最多的元素的次数
    max_count = counter1.most_common(1)[0][0]
    if len(counter1)>1 and max_count==0:
        max_count = counter1.most_common(2)[0][0]
    
    print(max_count)
    a=b+1
    g.append(max_count)

write_to_csv(g)








        


    
        

        

