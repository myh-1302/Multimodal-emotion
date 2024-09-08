

from model import BertTextModel_last_layer,BertTextModel_encode_layer
from utils import MyDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from config import parsers
import time
import os
import csv
def write_to_csv(preds):
        with open('predictions.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['bert+textcnn'])
            writer.writerows(preds)

def load_model(model_path, device, args):
    if args.select_model_last:
        model = BertTextModel_last_layer().to(device)
    else:
        model = BertTextModel_encode_layer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def text_class_name(texts, pred, args):
    preds=[]
    results = torch.argmax(pred, dim=1)
    results = results.cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    if len(results) != 1:
        for i in range(len(results)):
            print(f"文本：{texts[i]}\t预测的类别为：{classification_dict[results[i]]}")
            preds.append(results[i])
    else:
        print(f"文本：{texts}\t预测的类别为：{classification_dict[results[0]]}")
    return preds


def pred_one(args, model, device, start):
    tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
    text = "我们一起去打篮球吧！"
    encoded_pair = tokenizer(text, padding='max_length', truncation=True,  max_length=args.max_len, return_tensors='pt')
    token_ids = encoded_pair['input_ids']
    attn_masks = encoded_pair['attention_mask']
    token_type_ids = encoded_pair['token_type_ids']

    all_con = tuple(p.to(device) for p in [token_ids, attn_masks, token_type_ids])
    pred = model(all_con)
    text_class_name(text, pred, args)
    end = time.time()
    print(f"耗时为：{end - start} s")


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    root, name = os.path.split(args.save_model_last)
    save_best = os.path.join(root, str(args.select_model_last) + "_" +name)
    model = load_model(save_best, device, args)

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
    # pred_one(args, model, device, start)  # 预测一条文本
    x = MyDataset(texts, with_labels=False)
    xDataloader = DataLoader(x, batch_size=len(texts), shuffle=False)
    for batch_index, batch_con in enumerate(xDataloader):
        batch_con = tuple(p.to(device) for p in batch_con)
        pred = model(batch_con)
        
        preds=text_class_name(texts, pred, args)
    
    end = time.time()
    print(f"耗时为：{end - start} s")
    preds = [[pred] for pred in preds]
    write_to_csv(preds)
