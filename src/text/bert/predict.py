
from model import MyModel
from config import parsers
import torch
from transformers import BertTokenizer
import time
import csv
import pandas as pd

def write_to_csv(preds):
    # 读取已有的CSV文件
    df = pd.read_csv('predictions.csv')

    # 添加新的列
    df['bert'] = preds

    # 写入CSV文件
    df.to_csv('predictions.csv', index=False)

def load_model(device, model_path):
    myModel = MyModel().to(device)
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    return myModel


def process_text(text, bert_pred):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    mask = [1] * len(token_id) + [0] * (args.max_len + 2 - len(token_id))
    token_ids = token_id + [0] * (args.max_len + 2 - len(token_id))
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)
    x = torch.stack([token_ids, mask])
    return x


                    
def text_class_name(pred):
    result = torch.argmax(pred, dim=1)
    result = result.cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    print(f"文本：{text}\t预测的类别为：{classification_dict[result[0]]}")
    return result[0]
    
    
if __name__ == "__main__":
    # Convert integer values in preds to a list of lists
    
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_model(device, args.save_model_best)
    # 打开文件
    file = open("dataset\\text\\test.txt", "r", encoding="utf-8")

    # 按行读取文件内容
    lines = file.readlines()
    texts=[]
    # 处理每一行
    for line in lines:
        text, number = line.strip().split("\t")
        texts.append(text)
        # 现在，你可以使用text和number了

    # 关闭文件
    file.close()

    preds=[]

    print("模型预测结果：")
    for text in texts:
        x = process_text(text, args.bert_pred)
        with torch.no_grad():
            pred = model(x)
        
        d=text_class_name(pred)
        preds.append(int(d))
    end = time.time()
    
    write_to_csv(preds)
    print(f"耗时为：{end - start} s")
    
