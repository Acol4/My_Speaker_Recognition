#!/usr/bin/env python3
# train.py
import os,sys 

os.environ['HPCC_HOME'] = '/opt/hpcc'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/zion1/student/wyc/pykt-toolkit-main')
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm



from Config.config import Config
from data_process.data_loader import get_dataloader
from model.model import Classifier
from utils.utils import set_seed, get_cosine_schedule_with_warmup, model_fn, validate


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练说话人分类模型")
    parser.add_argument("--data_dir", type=str, default="./Dataset", help="数据目录路径")
    parser.add_argument("--model_path", type=str, default="model.ckpt", help="模型保存路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--n_workers", type=int, default=8, help="工作进程数")
    parser.add_argument("--total_steps", type=int, default=70000, help="总训练步数")
    parser.add_argument("--valid_steps", type=int, default=2000, help="验证步数间隔")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="学习率预热步数")
    parser.add_argument("--save_steps", type=int, default=10000, help="模型保存步数间隔")
    parser.add_argument("--seed", type=int, default=87, help="随机种子")
    
    return parser.parse_args()


def train():
    """训练主函数"""
    # 解析命令行参数
    #args = parse_args()
    
    # 创建配置对象
    config = Config()
    
    # 覆盖配置参数
    #config.data_dir = args.data_dir
    #config.model_path = args.model_path
    #config.batch_size = args.batch_size
    #config.n_workers = args.n_workers
    #config.total_steps = args.total_steps
    #config.valid_steps = args.valid_steps
    #config.warmup_steps = args.warmup_steps
    #config.save_steps = args.save_steps
    #config.seed = args.seed
    
    # 打印配置
    config.print_config()
    
    # 设置随机种子
    set_seed(config.seed)
    
    print(f"[Info]: 使用 {config.device} 进行训练")
    
    # 获取数据加载器
    train_loader, valid_loader, speaker_num = get_dataloader(
        config.data_dir, 
        config.batch_size, 
        config.n_workers,
        config.use_spec_augment,
        config.segment_len
    )
    
    print(f"[Info]: 数据集加载完成，共有 {speaker_num} 个说话人")
    print(f"[Info]: 训练集大小: {len(train_loader.dataset)}")
    print(f"[Info]: 验证集大小: {len(valid_loader.dataset)}")
    
    # 初始化模型
    model = Classifier(
        d_model=config.d_model,
        n_spks=speaker_num,
        
        nhead = config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        
        dropout=config.dropout,
        use_self_attention_pool=config.use_self_attention_pool
        
    ).to(config.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        config.warmup_steps, 
        config.total_steps
    )
    
    print(f"[Info]: 模型初始化完成")
    print(f"[Info]: 总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    best_accuracy = -1.0
    best_state_dict = None
    
    pbar = tqdm(total=config.valid_steps, ncols=0, desc="训练", unit=" 步")
    
    # 创建训练迭代器
    train_iterator = iter(train_loader)

    scaler = torch.amp.GradScaler('cuda')
    
    for step in range(config.total_steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        
        # 前向传播
        loss, accuracy = model_fn(batch, model, criterion, config.device)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # # 使用混合精度
        # with torch.amp.autocast('cuda'):
        #     loss, accuracy = model_fn(batch, model, criterion, config.device)
        
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # scheduler.step()
        # optimizer.zero_grad()
        
        # 更新进度条
        pbar.update()
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{accuracy.item():.4f}",
            step=step + 1,
        )
        
        # 验证
        if (step + 1) % config.valid_steps == 0:
            pbar.close()
            
            print(f"\n[Info]: 第 {step+1} 步开始验证...")
            valid_accuracy = validate(valid_loader, model, criterion, config.device)
            
            # 保存最佳模型
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()
                torch.save(best_state_dict, config.model_path)
                print(f"[Info]: 最佳模型已保存，准确率: {best_accuracy:.4f}")
            
            # 重新初始化进度条
            pbar = tqdm(total=config.valid_steps, ncols=0, desc="训练", unit=" 步")
        
        # 定期保存模型
        if (step + 1) % config.save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, f"model_step_{step+1}.ckpt")
            print(f"[Info]: 第 {step+1} 步模型已保存")
    
    pbar.close()
    print(f"\n[Info]: 训练完成!")
    print(f"[Info]: 最佳验证准确率: {best_accuracy:.4f}")
    
    # 保存最终模型
    if best_state_dict is not None:
        torch.save(best_state_dict, config.model_path)
        print(f"[Info]: 最终模型已保存到 {config.model_path}")


if __name__ == "__main__":
    train()