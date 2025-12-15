#!/usr/bin/env python3
# inference.py
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Config.config import Config
from data_process.data_loader import get_inference_dataloader
from model.model import Classifier
from utils.utils import save_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="说话人分类推理")
    parser.add_argument("--data_dir", type=str, default="./Dataset", help="数据目录路径")
    parser.add_argument("--model_path", type=str, default="model.ckpt", help="模型路径")
    parser.add_argument("--output_path", type=str, default="output.csv", help="输出文件路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--n_workers", type=int, default=8, help="工作进程数")
    
    return parser.parse_args()


def inference():
    """推理主函数"""
    # 解析命令行参数
    # args = parse_args()
    
    # 创建配置对象
    config = Config()
    
    # # 覆盖配置参数
    # config.data_dir = args.data_dir
    # config.model_path = args.model_path
    # config.output_path = args.output_path
    
    print(f"[Info]: 使用 {config.device} 进行推理")
    print(f"[Info]: 数据目录: {config.data_dir}")
    print(f"[Info]: 模型路径: {config.model_path}")
    print(f"[Info]: 输出路径: {config.output_path}")
    
    # 加载映射文件
    mapping_path = Path(config.data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    
    # 获取推理数据加载器
    dataloader = get_inference_dataloader(
        config.data_dir, 
        batch_size=1, 
        n_workers=config.n_workers,
    )
    
    print(f"[Info]: 数据加载完成，共 {len(dataloader.dataset)} 个样本")
    
    # 加载模型
    speaker_num = len(mapping["id2speaker"])
    model = Classifier(
        d_model=config.d_model, 
        n_spks=speaker_num,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        use_self_attention_pool=config.use_self_attention_pool
        
    ).to(config.device)
    
    model.load_state_dict(torch.load(config.model_path,weights_only=True))
    model.eval()
    
    print(f"[Info]: 模型加载完成，共 {speaker_num} 个说话人")
    print(f"[Info]: 开始推理...")
    
    # 推理
    results = [["Id", "Category"]]
    
    for feat_paths, mels in tqdm(dataloader, desc="推理"):
        with torch.no_grad():
            mels = mels.to(config.device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])
    
    # 保存结果
    save_results(results, config.output_path)
    
    # 打印统计信息
    print(f"\n[Info]: 推理完成!")
    print(f"[Info]: 处理了 {len(results)-1} 个样本")
    print(f"[Info]: 结果已保存到 {config.output_path}")


if __name__ == "__main__":
    inference()