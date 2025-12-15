# config.py
import torch
import sys,os 

class Config:
    """配置参数"""
    def __init__(self):
        # 数据配置
        self.data_dir = "./Dataset"
        self.batch_size = 32
        self.n_workers = 16
        self.segment_len = 128
        self.use_spec_augment = True

        # 模型配置
        self.d_model = 224
        self.nhead = 8
        self.dim_feedforward = 256
        self.dropout = 0.2
        self.num_layers = 3
        self.use_self_attention_pool = True
        

        # 训练配置
        self.seed = 87
        self.valid_steps = 1000
        self.warmup_steps = 5000
        self.save_steps = 10000
        self.total_steps = 70000
        self.learning_rate = 5e-4
        
        # 路径配置
        self.model_path = "model.ckpt"
        self.output_path = "output.csv"
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def print_config(self):
        """打印配置"""
        print("=" * 50)
        print("训练配置:")
        print(f"  数据目录: {self.data_dir}")
        print(f"  批量大小: {self.batch_size}")
        print(f"  工作进程: {self.n_workers}")
        print(f"  设备: {self.device}")
        print(f"  模型保存路径: {self.model_path}")
        print("=" * 50)