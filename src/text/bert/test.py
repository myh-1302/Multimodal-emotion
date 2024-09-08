

import torch
from tqdm import tqdm
from utils import read_data, MyDataset
from config import parsers
from torch.utils.data import DataLoader
from model import MyModel
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def max_iter(true_labels,predicted_labels):
    # 获取类别数量
    num_classes = max(max(true_labels), max(predicted_labels)) + 1

    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label][predicted_label] += 1

    # 类别标签
    labels = [f'Class {i}' for i in range(num_classes)]

    # 绘制混淆矩阵
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # 设置颜色条
    cbar = ax.figure.colorbar(im, ax=ax)

    # 设置标签
    ax.set(xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=labels, yticklabels=labels,
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label')

    # 在矩阵方格中显示数值
    thresh = confusion_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")

    # 自动调整布局
    fig.tight_layout()
    plt.savefig("D:\PycharmProjects\Multimodal emotion\output/text/混淆矩阵.jpg")
    # 显示图形
    plt.show()

def test_data():
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    test_text, test_label = read_data(args.test_file)
    test_dataset = MyDataset(test_text, test_label, args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MyModel().to(device)
    model.load_state_dict(torch.load(args.save_model_best))
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_text, batch_label in tqdm(test_dataloader):
            
            batch_label, batch_label = batch_label.to(device), batch_label.to(device)
            pred = model(batch_text)
            pred = torch.argmax(pred, dim=1)
            
            pred = pred.cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()
            
            all_pred.extend(pred)
            all_true.extend(label)
    with open(".\output/test.txt", "w") as file:
        for item1, item2, item3 in zip(test_text, test_label,all_pred):
            file.write(f"{item1}\t{item2}\t{item3}\n")
            
    accuracy = accuracy_score(all_true, all_pred)

    print(f"test dataset accuracy:{accuracy:.4f}")
    max_iter(all_true,all_pred)


if __name__ == "__main__":
    test_data()
