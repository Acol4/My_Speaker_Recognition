# utils.py
import math
import json
import random
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import csv


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[Info]: 随机种子设置为 {seed}")


def get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=0.5, 
    last_epoch=-1
):
    """带warmup的余弦学习率调度器"""
    def lr_lambda(current_step):
        # Warmup阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 余弦衰减阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion, device):
    """模型前向传播函数"""
    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)
    
    outs = model(mels)
    loss = criterion(outs, labels)
    
    # 计算准确率
    preds = outs.argmax(1)
    accuracy = torch.mean((preds == labels).float())
    
    return loss, accuracy


def validate(dataloader, model, criterion, device):
    """验证函数"""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="验证", unit=" 样本")
    
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
        
        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i+1):.2f}",
            accuracy=f"{running_accuracy / (i+1):.2f}",
        )
    
    pbar.close()
    model.train()
    
    return running_accuracy / len(dataloader)


def save_results(results, output_path):
    """保存结果到CSV文件"""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)
    print(f"[Info]: 结果已保存到 {output_path}")