import argparse
import functools

from mser.trainer import MSERTrainer
from mser.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'src\\audio\\LSTM\\bi_lstm.yml',      '配置文件')
add_arg("local_rank",       int,    0,                          '多卡训练需要的参数')
add_arg("use_gpu",          bool,   False,                       '是否使用GPU训练')
add_arg('save_model_path',  str,    'output/audio\model',                  '模型保存的路径')
add_arg('resume_model',     str,    None,                       '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    'model\TencentGameMate\chinese-hubert-base\pytorch_model.bin',                       '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MSERTrainer(configs=args.configs, use_gpu=args.use_gpu)

trainer.train(save_model_path=args.save_model_path,
              pretrained_model=args.pretrained_model)
