import pickle
import cv2
#加载视频帧
def load_video_frames(video_path,label,type,k):
    s=['negative','neutral','positive']
    frames = []
    cap = cv2.VideoCapture(video_path)
    #循环读取帧
    while cap.isOpened():
        ret, frame = cap.read()
        #读取完毕，跳出循环
        if not ret:
            break
        #将帧从BGR转换为RGB
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    #释放资源
    cap.release()
    
    #等距取15个图像
    step = int(len(frames) / 15)
    start = 0
    end = 15*step
    #索引
    indices = list(range(start, end, step))
    frames_selected = [frames [i] for i in indices]
    for frame in frames_selected:
        
        cv2.imwrite('.\dataset/video'+'\\'+type+'\\'+s[int(label)]+'\\image'+str(k)+'.jpg',frame)
        k=k+1
    return k
path='.\data\configure.pkl'
#打开文件 
with open(path, 'rb') as f:
    dataset = pickle.load(f)
#train
video_path_train=[]
for video_name in dataset['train']['id']:
    video_number = video_name.split('_')[1].split('$')[0]
    frame_number = video_name.split('_')[2].replace('$','')
    # 数据集路径列表
    video_path_train.append('.\data\Raw/video_{}/{}.mp4'.format(video_number, frame_number))
k=1
for i,name in enumerate(video_path_train):
    k=load_video_frames(name,dataset['train']['classification_labels_V'][i],'train',k) 

#vaild
video_path_valid=[]
for video_name in dataset['valid']['id']:
    video_number = video_name.split('_')[1].split('$')[0]
    frame_number = video_name.split('_')[2].replace('$','')
    # 数据集路径列表
    video_path_valid.append('.\data\Raw/video_{}/{}.mp4'.format(video_number, frame_number))
    
    

for i,name in enumerate(video_path_valid):
    k=load_video_frames(name,dataset['valid']['classification_labels_V'][i],'valid',k)
    
#test
video_path_test=[]
for video_name in dataset['test']['id']:
    video_number = video_name.split('_')[1].split('$')[0]
    frame_number = video_name.split('_')[2].replace('$','')
    # 数据集路径列表
    video_path_test.append('.\data\Raw/video_{}/{}.mp4'.format(video_number, frame_number))

for i,name in enumerate(video_path_test):
    k=load_video_frames(name,dataset['test']['classification_labels_V'][i],'test',k)



