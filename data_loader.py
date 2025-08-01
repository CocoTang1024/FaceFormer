import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args):
    # 1️⃣ 先打印提示
    print("Loading data...")

    # 2️⃣ 用嵌套字典存单个文件的信息：data[file_key]['audio'/'vertice'/...]
    data = defaultdict(dict)

    # 3️⃣ 三个列表分别存训练 / 验证 / 测试样本（元素就是上面的 data[key] 字典）
    train_data, valid_data, test_data = [], [], []

    # 4️⃣ 组装音频和顶点（面部网格）文件夹的绝对路径
    audio_path    = os.path.join(args.dataset, args.wav_path)        # 例：vocaset/wav
    vertices_path = os.path.join(args.dataset, args.vertices_path)   # 例：vocaset/vertices_npy

    # 5️⃣ 准备 Wav2Vec2 的处理器（把原始 wave 转成模型输入）
    processor = Wav2Vec2Processor.from_pretrained(
        os.environ["WAV2VEC_PATH"]  # 你也可以直接写死路径
    )

    # 6️⃣ 读取每个说话人的“个人模板”（静态人脸）
    template_file = os.path.join(args.dataset, args.template_file)   # 例：vocaset/templates.pkl
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    # ------ 开始扫描音频文件夹 ------
    print(f"Scanning audio path: {audio_path}")
    print(f"Train subjects: {args.train_subjects}")

    for r, ds, fs in os.walk(audio_path):          # 7️⃣ 遍历目录下所有文件
        for f in tqdm(fs):                         # tqdm 进度条
            if f.endswith("wav"):                  # 只处理 .wav 音频
                print(f"Processing file: {f}")

                wav_path = os.path.join(r, f)                       # 8️⃣ 音频绝对路径
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)  # 固定采样率 16k
                # 9️⃣ 用 Wav2Vec2Processor 把波形转成特征向量
                input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                
                key = f.replace("wav", "npy")      # 10️⃣ 约定：音频 key 与顶点文件名一致，只后缀不同
                data[key] = {}                     # 创建条目

                # 11️⃣ 存音频特征
                data[key]["audio"] = input_values

                # 12️⃣ 抽取说话人 ID，例如 FaceTalk_170728_03272_TA
                subject_id = "_".join(key.split("_")[:-1])
                print(f"Subject ID extracted: {subject_id}")

                # 13️⃣ 若模板里找不到此人，就跳过这个样本
                if subject_id not in templates:
                    print(f"Warning: No template found for subject {subject_id}")
                    del data[key]
                    continue

                # 14️⃣ 写入人脸模板并展平成一维
                temp = templates[subject_id]
                data[key]["name"]     = f          # 原始文件名
                data[key]["template"] = temp.reshape((-1))

                # 15️⃣ 找对应顶点序列 .npy
                vertice_path = os.path.join(vertices_path, f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    print(f"Warning: No vertex file found for {f}")
                    del data[key]
                    continue
                else:
                    # 16️⃣ 根据数据集不同，做不同的读取/下采样策略
                    if args.dataset == "vocaset":
                        # vocaset 顶点太大，取隔帧 (每 2 帧 1 取) 减少内存
                        data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)[::2, :]
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)

    # ------ 按说话人拆分 train/val/test ------
    subjects_dict = {
        "train": [i for i in args.train_subjects.split(" ")],
        "val"  : [i for i in args.val_subjects.split(" ")],
        "test" : [i for i in args.test_subjects.split(" ")]
    }

    # 17️⃣ 每个数据集有各自的句子编号划分规则
    splits = {
        'vocaset': {
            'train': range(1, 41),   # 前 1~40 句可做训练
            'val'  : range(21, 41),  # 21~40 句既能当验证也能当测试
            'test' : range(21, 41)
        },
        'BIWI': {
            'train': range(1, 33),
            'val'  : range(33, 37),
            'test' : range(37, 41)
        }
    }

    # ------ 遍历 data，将样本放进对应集合 ------
    for k, v in data.items():
        subject_id  = "_".join(k.split("_")[:-1])      # 说话人
        sentence_id = int(k.split(".")[0][-2:])        # 句子序号（文件名最后两位数字）

        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)

        if subject_id in subjects_dict["val"]   and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)

        if subject_id in subjects_dict["test"]  and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    # 18️⃣ 打印各集合样本数
    print(len(train_data), len(valid_data), len(test_data))

    # 19️⃣ 返回三组数据 + 说话人字典
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data,subjects_dict,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset

if __name__ == "__main__":
    get_dataloaders()
    