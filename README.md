
# 说话人识别系统

这是一个基于Transformer的说话人识别系统，能够从语音片段中识别出说话人的身份。

## 项目结构

```
.
├── Config          # 配置文件目录
├── Dataset         # 数据集目录（需自行准备）
├── Inference       # 推理代码目录
├── data_process    # 数据处理模块
├── model           # 模型定义
├── train           # 训练代码
├── utils           # 工具函数
├── main.py         # 主程序入口
├── environment.yml # 环境配置文件
└── README.md       # 说明文档
```

## 环境配置

### 使用Conda安装依赖

## 数据集

项目使用预处理后的梅尔频谱图作为输入数据，数据组织方式如下：
- `Dataset/mapping.json`: 说话人ID与名称的映射关系
- `Dataset/metadata.json`: 训练数据元信息
- `Dataset/testdata.json`: 测试数据列表

## 使用方法

### 1. 训练模型

```bash
python train/train.py
```

训练参数可在 [Config/config.py](./Config/config.py) 中修改，包括：
- batch_size: 批次大小
- learning_rate: 学习率
- total_steps: 总训练步数
- d_model: 模型维度
- nhead: 注意力头数

训练过程中会定期验证并保存最佳模型。

### 2. 运行推理

```bash
python Inference/Inference.py
```

推理脚本会加载训练好的模型对测试数据进行预测，并将结果保存到 `output.csv` 文件中。


## 输出格式

推理结果将以CSV格式保存，包含两列：
- Id: 样本标识符
- Category: 预测的说话人类别
