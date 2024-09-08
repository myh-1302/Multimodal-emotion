import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_audio_features(audio_file_path):
    """
    从音频文件中提取特征。

    参数：
    audio_file_path (str): 音频文件的路径。

    返回值：
    features (ndarray): 提取的音频特征矩阵。
    """

    # 加载音频文件
    audio, sr = librosa.load(audio_file_path, sr=22050)

    # 提取 log F0
    f0 = librosa.yin(audio, fmin=50, fmax=300, sr=sr)
    f0 = f0.reshape(1,-1)
    print(f0.shape)
    
    # 提取 MFCC
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20)
    print(mfcc.shape)
    # 提取 CQT
    cqt = librosa.cqt(audio, sr=sr, n_bins=12)
    print(cqt.shape)
    # 拼接所有特征
    features = np.concatenate((np.log(f0), mfcc, cqt), axis=0)

    # 转置特征矩阵
    features = features.T

    return features

def plot_audio_waveform(audio_file_path):
    """
    绘制音频波形图。

    参数：
    audio_file_path (str): 音频文件的路径。
    """

    # 加载音频文件
    audio, sr = librosa.load(audio_file_path, sr=22050)

    # 创建时间轴
    duration = len(audio) / sr
    time = np.linspace(0, duration, len(audio))

    # 绘制波形图
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.show()

import librosa.display

# 调用函数绘制音频波形图
audio_file_path = 'dataset\\audio\\train\\negative\\audio1.wav'
plot_audio_waveform(audio_file_path)

# 提取音频特征
features = extract_audio_features(audio_file_path)
print(features.shape)
# 可视化特征
plt.figure(figsize=(10, 4))
librosa.display.specshow(features.T, x_axis='time')
plt.colorbar()
plt.title('Audio Features')
plt.show()

