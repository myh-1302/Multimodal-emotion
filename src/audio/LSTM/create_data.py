import os

from mser.trainer import MSERTrainer
import random
def luan(path):
    # 读取文本文件内容
    with open(path, 'r') as file:
        lines = file.readlines()

    # 打乱行的顺序
    random.shuffle(lines)
    # 将打乱后的行写回到文本文件
    with open(path, 'w') as file:
        file.writelines(lines)
    # 关闭文件
    file.close()
# 生成数据列表
def get_data_list(audio_dir, list_path):
    sound_sum = 0

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')
    f_valid = open(os.path.join(list_path, 'valid_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8')
    i=0
    labels = ["negative", "neutral", "positive"]
    for label in labels:
        f_label.write(f'{label}\n')
        
        train_dir = os.path.join(audio_dir, 'train', label)
        test_dir = os.path.join(audio_dir, 'test', label)
        valid_dir = os.path.join(audio_dir, 'valid', label)

        train_files = os.listdir(train_dir)
        test_files = os.listdir(test_dir)
        valid_files = os.listdir(valid_dir)

        for file in train_files:
            sound_path = os.path.join(train_dir, file).replace('\\', '/')
            f_train.write(f'{sound_path}\t{i}\n')
            sound_sum += 1

        for file in test_files:
            sound_path = os.path.join(test_dir, file).replace('\\', '/')
            f_test.write(f'{sound_path}\t{i}\n')
            sound_sum += 1

        for file in valid_files:
            sound_path = os.path.join(valid_dir, file).replace('\\', '/')
            f_valid.write(f'{sound_path}\t{i}\n')
            sound_sum += 1
        i+=1
    f_label.close()
    f_train.close()
    f_test.close()
    f_valid.close()
    luan(os.path.join(list_path, 'train_list.txt'))
    luan(os.path.join(list_path, 'valid_list.txt'))
    luan(os.path.join(list_path, 'test_list.txt'))
    




# 生成归一化文件
def create_standard(config_file):
    trainer = MSERTrainer(configs=config_file,use_gpu=False)
    trainer.get_standard_file()


if __name__ == '__main__':
    # get_data_list('dataset/audios', 'dataset')
    get_data_list('dataset/audio', 'dataset/audio')
    create_standard('src/audio/bi_lstm.yml') 
