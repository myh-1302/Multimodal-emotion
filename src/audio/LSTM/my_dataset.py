import os
import numpy as np
import datasets
import soundfile as sf
import torch

class MyDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    'audio': {
                        'path': datasets.Value('string'),
                        'array': datasets.Array1D(dtype='float32',sampling_rate=16000),
                        'sampling_rate': datasets.Value('int32')
                    },
                    'label': datasets.ClassLabel(names=['negative', 'neutral', 'positive'])
                }
            ),
            supervised_keys=('audio', 'label'),
        )

    def _split_generators(self, dl_manager):
        data_dir ='dataset/audio'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'data_dir': data_dir, 'split': 'train'}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'data_dir': data_dir, 'split': 'test'}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={'data_dir': data_dir, 'split': 'valid'}
            ),
        ]

    def _generate_examples(self, data_dir, split):
        split_dir = os.path.join(data_dir, split)
        labels = ['negative', 'neutral', 'positive']
        label_idx = 0  # Assuming the label index for "electronic" is 0
        label = labels[label_idx]
        label_dir = os.path.join(split_dir, label)
        wav_files = os.listdir(label_dir)
        for wav_file in wav_files:
            file_path = os.path.join(label_dir, wav_file)
            print(file_path)
            audio_data, _ = sf.read(file_path)  # 使用soundfile库加载音频数据，不获取采样率
            sampling_rate = 16000  # 自定义采样率，这里假设为44100
            yield {
                'audio': {
                    'path': file_path,
                    'array': audio_data,
                    'sampling_rate': sampling_rate
                },
                'label': label_idx
            }