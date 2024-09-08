
import os
import datasets
def generate_examples(data_dir, split):
    split_dir = os.path.join(data_dir, split)
    labels = ['negative', 'neutral', 'positive']
    for label in labels:
        label_dir = os.path.join(split_dir, label)
        wav_files = os.listdir(label_dir)
        for wav_file in wav_files:
            file_path = os.path.join(label_dir, wav_file)
            yield file_path, {'label': label}
class MyDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    'file_path': datasets.Value('string'),
                    'label': datasets.ClassLabel(names=['negative', 'neutral', 'positive'])
                }
            ),
            supervised_keys=('file_path', 'label'),
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
        yield from generate_examples(data_dir, split)
