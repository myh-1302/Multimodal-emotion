from tqdm import tqdm
import torch
import torch.nn.functional as F
import soundfile as sf
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
from transformers import TrainingArguments
import util
from transformers import Trainer
import torch
import pickle
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from datasets import Dataset, Features, ClassLabel
from datasets import Dataset, Audio
import os
from datasets import Features
""" def import_audio_dataset(data_dir, type_data):
    # 定义类别列表
    categories = ["negative", "neutral", "positive"]

    # 导入数据集
    def import_data(split):
        files = []
        labels = []
        for idx, category in enumerate(categories):
            category_dir = os.path.join(data_dir, split, category)
            category_files = [os.path.join(category_dir, file) for file in os.listdir(category_dir) if file.endswith(".wav")]
            files.extend(category_files)
            labels.extend([idx] * len(category_files))  # 添加相应的标签
        return files, labels

    # 创建包含音频路径和标签的字典
    audio_files, audio_labels = import_data(type_data)
    audio_dict = {"audio": audio_files, "label": torch.tensor(audio_labels)}

    # 创建数据集对象
    audio_dataset = Dataset.from_dict(audio_dict)

    # 设置列类型
    audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))

    return audio_dataset
train_dataset=import_audio_dataset("dataset/audio",'train')
validation_dataset=import_audio_dataset("dataset/audio",'valid')


model_name_or_path = "xmj2002/hubert-base-ch-speech-emotion-recognition"


print(train_dataset) """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import random

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel

""" model_name_or_path = "xmj2002/hubert-base-ch-speech-emotion-recognition"
duration = 6
sample_rate = 16000

config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
)




class HubertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()

    def forward(self, x):
        outputs = self.hubert(x)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x

from datasets import load_dataset, Audio
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

minds = load_dataset("audiofolder", data_dir="dataset/audio")
print(minds['train']['label'])



labels = minds["train"].features["label"].names

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

num_labels=len(label2id)

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
def preprocess_function(examples):
    audio_arrays = [np.array(x["array"], dtype=np.float32) for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate,padding=True, max_length=16000, truncation=True
    )
    return inputs
encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
print(encoded_minds['test'].features['label'])
     
import evaluate
accuracy = evaluate.load("accuracy")
import numpy as np


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)
print(model.config)
training_args = TrainingArguments(
    output_dir="output/audio\model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train() """


import torch
import torch.nn.functional as F
import soundfile as sf
#from fairseq import checkpoint_utils

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

model_path="model\TencentGameMate\chinese-hubert-base"
wav_path="dataset/audio/test/negative/audio1825.wav"
mask_prob=0.0
mask_length=10

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
#model = Wav2Vec2Model.from_pretrained(model_path)

# for pretrain: Wav2Vec2ForPreTraining
model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

model = model.to(device)
model = model.half()
model.eval()

wav, sr = sf.read(wav_path)
input_values = feature_extractor(wav, return_tensors="pt").input_values
input_values = input_values.half()
input_values = input_values.to(device)

# for Wav2Vec2ForPreTraining
batch_size, raw_sequence_length = input_values.shape
sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.0, mask_length=2)
mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)

with torch.no_grad():
    #outputs = model(input_values)
    #last_hidden_state = outputs.last_hidden_state

    # for Wav2Vec2ForPreTraining
    outputs = model(input_values, mask_time_indices=mask_time_indices, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]





