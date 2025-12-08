# data_loader.py
import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


def clean_json_file(file_path):
    """清理JSON文件中的NUL字符和其他无效字符"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 移除NUL字符（\x00）和其他控制字符（除了常见的如\n, \t, \r）
        # 这里使用正则表达式移除所有ASCII控制字符（除了常见的空白字符）
        import re
        # 移除除\t,\n,\r之外的所有控制字符
        cleaned_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        # 如果内容被修改了，重新写入文件
        if cleaned_content != content:
            print(f"Cleaned NUL characters from {file_path}")
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_content)
        
        return cleaned_content
    except Exception as e:
        print(f"Error cleaning file {file_path}: {e}")
        # 如果清理失败，尝试读取原始内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()


class SpeakerDataset(Dataset):
    """训练数据集类"""
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len
        
        # 加载speaker映射
        mapping_path = Path(data_dir) / "mapping.json"
        # 先清理文件
        mapping_content = clean_json_file(mapping_path)
        mapping = json.loads(mapping_content)
        self.speaker2id = mapping["speaker2id"]
        
        # 加载元数据
        metadata_path = Path(data_dir) / "metadata.json"
        # 清理metadata.json文件
        metadata_content = clean_json_file(metadata_path)
        metadata = json.loads(metadata_content)["speakers"]
        
        # 获取speaker数量
        self.speaker_num = len(metadata.keys())
        self.data = []
        
        # 构建数据列表
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        
        # 加载mel-spectrogram
        mel = torch.load(os.path.join(self.data_dir, feat_path),weights_only=True)
        
        # 分段处理
        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker
    
    def get_speaker_number(self):
        return self.speaker_num


class InferenceDataset(Dataset):
    """推理数据集类"""
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        # 清理testdata.json文件
        metadata_content = clean_json_file(testdata_path)
        metadata = json.loads(metadata_content)
        self.data_dir = data_dir
        self.data = metadata["utterances"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path),weights_only=True)
        
        return feat_path, mel


def collate_batch(batch):
    """训练数据collate函数"""
    mel, speaker = zip(*batch)
    # 填充mel-spectrogram
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    return mel, torch.FloatTensor(speaker).long()


def inference_collate_batch(batch):
    """推理数据collate函数"""
    feat_paths, mels = zip(*batch)
    return feat_paths, torch.stack(mels)


def get_dataloader(data_dir, batch_size, n_workers, segment_len=128):
    """生成训练和验证数据加载器"""
    dataset = SpeakerDataset(data_dir, segment_len)
    speaker_num = dataset.get_speaker_number()
    
    # 分割数据集
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)
    
    # 训练数据加载器
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=True,
    )
    
    # 验证数据加载器
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    
    return train_loader, valid_loader, speaker_num


def get_inference_dataloader(data_dir, batch_size=1, n_workers=8):
    """生成推理数据加载器"""
    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        collate_fn=inference_collate_batch,
    )
    return dataloader