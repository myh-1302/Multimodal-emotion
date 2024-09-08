import pandas as pd
import pandas as pd
import pickle

def write_to_csv(preds):
    # 读取已有的CSV文件
    df = pd.read_csv('predictions.csv')

    # 添加新的列
    df['true_aiduo'] = preds

    # 写入CSV文件
    df.to_csv('predictions.csv', index=False)
    
def read_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data
import numpy as np
data=read_from_pkl('data\\configure.pkl')
# 假设data是你的浮点数数组
data = np.array(data['test']['classification_labels_A'])

# 使用astype函数将浮点数数组转换为整数数组
data = data.astype(int)

print(data)
write_to_csv(data)