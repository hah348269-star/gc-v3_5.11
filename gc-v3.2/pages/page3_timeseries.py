# pages/page3_timeseries.py
# 模块三：LSTM 时间序列碳排放预测 + 异常预警
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据集类
class EnergyTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=24, pred_length=1):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length:idx+self.seq_length+self.pred_length, 0]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# LSTM 模型
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1, pred_length=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_dim, pred_length)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def show_page3():
    st.header("⏱️ 模块三：LSTM 时间序列碳排放预测与异常预警")
    st.markdown("---")

    if "df" not in st.session_state or len(st.session_state.df) == 0:
        st.warning("请先在模块一加载数据！")
        return

    df = st.session_state.df.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================== 1. 数据准备 ====================
    st.subheader("1. 时间序列数据构建（24小时输入 → 预测1小时）")
    SEQ_LENGTH = 24
    PRED_LENGTH = 1

    # ----------------- 这里改成你的列名 -----------------
    if 'daily_emission' not in df.columns:
        st.error("请确保数据包含 daily_emission 列")
        return

    # 补全特征
    for c in ['temperature_c','humidity_pct','electricity_kwh','occupancy_rate']:
        if c not in df.columns:
            df[c] = np.random.randn(len(df))

    # 时间特征
    df['datetime'] = pd.to_datetime(df['date'])
    df['hour'] = np.random.randint(0,24,len(df))
    df['weekday'] = df['datetime'].dt.weekday
    df['is_weekend'] = (df['weekday']>=5).astype(int)

    # ----------------- 这里改成你的列名 -----------------
    feature_cols = [
        'daily_emission',
        'temperature_c','humidity_pct',
        'electricity_kwh','occupancy_rate',
        'hour','weekday','is_weekend'
    ]
    data = df[feature_cols].values

    # 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 数据太少时自动降低序列长度
    if len(data_scaled) < 50:
        SEQ_LENGTH = 5

    # 划分
    train_size = max(int(len(data_scaled)*0.7), SEQ_LENGTH+1)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    try:
        train_dataset = EnergyTimeSeriesDataset(train_data, SEQ_LENGTH, PRED_LENGTH)
        test_dataset = EnergyTimeSeriesDataset(test_data, SEQ_LENGTH, PRED_LENGTH)
        st.success(f"✅ 构建完成：训练集{len(train_dataset)} / 测试集{len(test_dataset)}")
    except:
        st.error("数据量太少，无法构建时间序列，请补充更多数据！")
        return

    # ==================== 2. 模型训练 ====================
    st.markdown("---")
    st.subheader("2. LSTM 模型训练")
    if st.button("启动 LSTM 训练"):
        if len(train_dataset) == 0:
            st.error("训练集为空！")
            return

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        input_dim = len(feature_cols)
        model = SimpleLSTMModel(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        bar = st.progress(0)
        for e in range(10):
            model.train()
            for x,y in train_loader:
                x,y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            bar.progress((e+1)/10)

        # 测试集预测
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x,y in test_loader:
                pred = model(x.to(device)).cpu().numpy()
                preds.extend(pred.flatten())
                trues.extend(y.numpy().flatten())

        # 反归一化
        def inverse_transform(arr):
            f = np.zeros((len(arr), scaler.n_features_in_))
            f[:,0] = arr
            return scaler.inverse_transform(f)[:,0]

        preds_orig = inverse_transform(np.array(preds))
        trues_orig = inverse_transform(np.array(trues))

        # 指标
        rmse = np.sqrt(mean_squared_error(trues_orig, preds_orig))
        mae = mean_absolute_error(trues_orig, preds_orig)

        st.session_state.ts_model = model
        st.session_state.trues = trues_orig
        st.session_state.preds = preds_orig
        st.session_state.ts_metrics = {"RMSE":rmse,"MAE":mae}

        st.success("训练完成！")
        col1, col2 = st.columns(2)
        col1.metric("测试集 RMSE", f"{rmse:.2f}")
        col2.metric("测试集 MAE", f"{mae:.2f}")

    # ==================== 3. 预测曲线 + 异常曲线 左右并排 ====================
    if "trues" in st.session_state:
        st.markdown("---")
        st.subheader("3. 预测曲线 & 异常预警")

        # ========== 核心修改：左右并排布局 ==========
        col_left, col_right = st.columns(2)

        # 左图：预测曲线
        with col_left:
            fig1, ax1 = plt.subplots(figsize=(5, 3.5))
            ax1.plot(st.session_state.trues[:100], label='真实值')
            ax1.plot(st.session_state.preds[:100], label='预测值', alpha=0.7)
            ax1.set_title("LSTM 碳排放预测")
            ax1.legend()
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)

        # 右图：异常预警曲线
        with col_right:
            errors = np.abs(st.session_state.trues - st.session_state.preds)
            threshold = np.mean(errors) + 2 * np.std(errors)
            anomalies = errors > threshold

            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.plot(errors[:100], label='预测误差', color='#ff6666')
            ax2.axhline(y=threshold, color='red', linestyle='--', label='异常阈值')
            ax2.set_title("碳排放异常预警")
            ax2.legend()
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

        # ========== 指标展示保持不变 ==========
        st.markdown("---")
        st.subheader("4. 异常碳排放预警")
        st.metric("异常阈值", f"{threshold:.2f}")
        st.metric("检测到异常点数", np.sum(anomalies))

        if np.sum(anomalies) > 0:
            st.error("🔴 发现异常碳排放时段！请检查能耗设备")
        else:
            st.success("🟢 碳排放正常")

        # ==================== 5. 导出 ====================
        st.markdown("---")
        st.subheader("5. 导出预测报告")
        res_df = pd.DataFrame({
            "真实碳排放": st.session_state.trues,
            "预测碳排放": st.session_state.preds,
            "误差": errors,
            "是否异常": anomalies
        })
        csv = res_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("下载预测结果", csv, "lstm_prediction.csv")

if __name__ == "__main__":
    show_page3()