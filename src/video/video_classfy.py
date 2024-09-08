from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import pickle
import os
import shutil
import pytorchvideo.data
import imageio
import numpy as np
from IPython.display import Image
import evaluate
import torch
from transformers import TrainingArguments, Trainer
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
#模型
all_video_file_paths=[]
input_directory='./dataset/video'
for root, dirs, files in os.walk(input_directory):
    for file in files:
        # 获取文件的完整路径
        file_path = os.path.join(root, file)
        
        all_video_file_paths.append(file_path)
class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}



model_ckpt = "D:\PycharmProjects\Multimodal emotion\model/videomae-base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

#处理数据
mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames

sample_rate = 2
fps = 10
clip_duration = num_frames_to_sample * sample_rate / fps
print("clip_duration: ", clip_duration)
train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=224, max_size=224),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)
def get_path(type):
    s=['negative','neutral','positive']
    dataset_root_path='./dataset/video'
    video_train_paths=[]
    label_dict=[]
    input_directory='./dataset/video/'+type
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            label = file_path.split("\\")[-2]
            if label == 'negative':
                l=0
            elif label == 'neutral':
                l=1
            elif label == 'positive':
                l=2
            #print(label)
            labels={'label':l}
            label_dict.append(labels)
            #print(labels)
            video_train_paths.append(file_path)
    return list(zip(video_train_paths, label_dict))

paths_train = get_path('train')
train_dataset = pytorchvideo.data.LabeledVideoDataset(
    labeled_video_paths=paths_train,
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)

val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

paths_valid = get_path('valid')
val_dataset = pytorchvideo.data.LabeledVideoDataset(
    labeled_video_paths=paths_valid,
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)
paths_test = get_path('test')
test_dataset = pytorchvideo.data.LabeledVideoDataset(
    labeled_video_paths=paths_test,
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)
print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)


def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)

def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.
    
    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename

def display_gif(video_tensor, gif_name="sample.gif"):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)
def save_gif(video_tensor, filename="sample.gif"):
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, filename)
    return gif_filename
sample_video = next(iter(train_dataset))
video_tensor = sample_video["video"]
gif_filename = save_gif(video_tensor)
print(f"Saved GIF to {gif_filename}")



#训练模型
batch_size=10

num_epochs = 4

args = TrainingArguments(
    output_dir='./runs',
    logging_dir='./logs',           
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
)


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
trainer = Trainer(         
    model=model,
    args=args, 
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    
)
train_results = trainer.train()
