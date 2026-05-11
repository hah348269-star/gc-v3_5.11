# -*- coding: utf-8 -*-
"""
全局配置文件
适配：core / utils / pages 所有模块
"""
import os

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROC_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_SAVE_DIR = os.path.join(DATA_DIR, "models")

# 日志路径
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 报告输出路径
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# 可视化图表保存路径
FIG_SAVE_DIR = os.path.join(BASE_DIR, "figures")

# 自动创建文件夹
for path in [DATA_DIR, RAW_DATA_DIR, PROC_DATA_DIR, MODEL_SAVE_DIR,
             LOG_DIR, REPORT_DIR, FIG_SAVE_DIR]:
    os.makedirs(path, exist_ok=True)

# ===================== 数据库配置 =====================
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "asdfghjkl77520@YYQ",
    "database": "my_info",
    "charset": "utf8mb4"
}

# ===================== LLM 配置 =====================
LLM_CONFIG = {
    "llm_type": "openai",
    "api_key": "your_api_key",
    "model_name": "gpt-4",
    "temperature": 0.7
}

# ===================== 政策匹配配置 =====================
POLICY_CONFIG = {
    "top_k": 5,
    "similarity_threshold": 0.5
}

# ===================== 模型训练通用配置 =====================
TRAIN_CONFIG = {
    "random_seed": 42,
    "test_size": 0.2,
    "batch_size": 32,
    "epochs": 50
}

# ===================== LSTM 时序预测配置 =====================
LSTM_CONFIG = {
    "time_step": 24,
    "hidden_size": 64,
    "learning_rate": 0.001
}

# ===================== PPO 强化学习配置 =====================
PPO_CONFIG = {
    "state_dim": 11,
    "action_dim": 4,
    "gamma": 0.99,
    "lr": 3e-4
}

# ===================== 异常检测配置 =====================
ANOMALY_CONFIG = {
    "std_threshold": 3.0,
    "window_size": 24
}