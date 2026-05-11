# pages/page2_predict.py
# 模块二：碳排放预测与降碳潜力评估
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def show_page2():
    st.header("📈 模块二：碳排放预测与降碳潜力评估")
    st.markdown("---")

    if "df" not in st.session_state or len(st.session_state.df) == 0:
        st.warning("请先在模块一加载数据！")
        return

    df = st.session_state.df
    df_full = df.copy()

    # ==================== 1. 特征工程 ====================
    st.subheader("1. 特征工程（20维）")
    with st.expander("查看特征说明", expanded=False):
        st.code("""
特征包含：时间、气象、建筑、能耗、光伏、人员密度
hour, weekday, is_weekend, month, season
temperature_c, humidity_pct, wind_speed_ms, solar_radiation_wm2
area_sqm, build_year, building_type, hvac_type, insulation_level
electricity_kwh, gas_m3, heating_kwh, cooling_kwh, pv_generation_kwh, occupancy_rate
        """)

    # 补全必要列（兼容真实/模拟数据）
    for col in ['temperature_c', 'humidity_pct', 'wind_speed_ms', 'solar_radiation_wm2',
                'area_sqm', 'build_year', 'building_type', 'hvac_type', 'insulation_level',
                'electricity_kwh', 'gas_m3', 'heating_kwh', 'cooling_kwh',
                'pv_generation_kwh', 'occupancy_rate', 'building_id']:
        if col not in df_full.columns:
            df_full[col] = np.random.randn(len(df_full))

    # 时间特征
    df_full['datetime'] = pd.to_datetime(df_full['date'])
    df_full['hour'] = np.random.randint(0, 24, len(df_full))
    df_full['weekday'] = df_full['datetime'].dt.weekday
    df_full['month'] = df_full['datetime'].dt.month
    df_full['is_weekend'] = (df_full['weekday'] >= 5).astype(int)
    df_full['season'] = df_full['month'].apply(lambda x:
                                               1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7,
                                                                                                             8] else 4)

    # 类别编码
    df_full['building_type_encoded'] = df_full['building_type'].astype('category').cat.codes
    df_full['hvac_type_encoded'] = df_full['hvac_type'].astype('category').cat.codes
    df_full['insulation_level_encoded'] = df_full['insulation_level'].astype('category').cat.codes

    # 特征列表
    feature_cols = [
        'hour', 'weekday', 'is_weekend', 'month', 'season',
        'temperature_c', 'humidity_pct', 'wind_speed_ms', 'solar_radiation_wm2',
        'area_sqm', 'build_year', 'building_type_encoded', 'hvac_type_encoded', 'insulation_level_encoded',
        'electricity_kwh', 'gas_m3', 'heating_kwh', 'cooling_kwh', 'pv_generation_kwh', 'occupancy_rate'
    ]

    X = df_full[feature_cols]
    y = df_full['daily_emission']

    st.success(f"✅ 特征构造完成：共 {len(feature_cols)} 维特征")

    # ==================== 2. 数据集划分 ====================
    st.markdown("---")
    st.subheader("2. 数据集划分 7:1.5:1.5")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    col1, col2, col3 = st.columns(3)
    col1.metric("训练集", f"{len(X_train)} 条")
    col2.metric("验证集", f"{len(X_val)} 条")
    col3.metric("测试集", f"{len(X_test)} 条")

    # ==================== 3. 训练模型 ====================
    st.markdown("---")
    st.subheader("3. 线性回归模型训练")
    if st.button("开始训练模型"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)

        # 评估指标
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        mae = mean_absolute_error(y_test, pred_test)
        mape = np.mean(np.abs((y_test - pred_test) / (y_test + 1e-10))) * 100
        r2 = r2_score(y_test, pred_test)

        st.session_state.pred_model = model
        st.session_state.pred_test = pred_test
        st.session_state.y_test = y_test
        st.session_state.metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

        st.success("训练完成！")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{rmse:.2f}")
        c2.metric("MAE", f"{mae:.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")
        c4.metric("R²", f"{r2:.3f}")

    # ==================== 4. 预测可视化 ====================
    if "y_test" in st.session_state:
        st.markdown("---")
        st.subheader("4. 预测效果")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(st.session_state.y_test, st.session_state.pred_test, alpha=0.5)
        ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--')
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.set_title("线性回归预测效果")
        st.pyplot(fig)

    # ==================== 5. 降碳潜力评估 ====================
    st.markdown("---")
    st.subheader("5. 降碳潜力评估（P20基准）")
    if st.button("计算降碳潜力"):
        model = st.session_state.get("pred_model", None)
        if model is None:
            st.error("请先训练模型！")
            return

        X_full = df_full[feature_cols]
        pred_full = model.predict(X_full)
        df_full["predicted_carbon"] = pred_full

        # ==================== ✅ 修复：基准改用建筑名称计算 ====================
        if "building_name" in df_full.columns:
            # 按建筑分组计算基准
            df_full["benchmark_carbon"] = df_full.groupby("building_name")["predicted_carbon"].transform(lambda x: x.quantile(0.2))
        else:
            # 没有建筑名就按整体
            df_full["benchmark_carbon"] = df_full["predicted_carbon"].quantile(0.2)

        # ==================== ✅ 修复：降碳潜力公式 ====================
        df_full["carbon_potential"] = df_full["predicted_carbon"] - df_full["benchmark_carbon"]
        df_full["carbon_potential"] = df_full["carbon_potential"].clip(lower=0)  # 最小为0

        # 分级
        q50 = df_full["carbon_potential"].quantile(0.5)
        q80 = df_full["carbon_potential"].quantile(0.8)
        df_full["potential_level"] = df_full["carbon_potential"].apply(
            lambda x: "高" if x >= q80 else "中" if x >= q50 else "低"
        )

        # 汇总（按建筑名称，不是building_id）
        group_col = "building_name" if "building_name" in df_full.columns else "building_id"
        summary = df_full.groupby(group_col).agg({
            "predicted_carbon": "mean",
            "benchmark_carbon": "mean",
            "carbon_potential": "mean",
            "potential_level": lambda x: x.mode()[0]
        }).reset_index()

        st.session_state.potential = df_full
        st.session_state.summary = summary

        st.dataframe(summary.head(10), use_container_width=True)
        st.success("✅ 降碳潜力评估完成！")

    # ==================== 6. 导出 ====================
    if "summary" in st.session_state:
        st.markdown("---")
        st.subheader("6. 导出报告")
        csv = st.session_state.summary.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("下载降碳潜力评估表", csv, "降碳潜力评估.csv")


if __name__ == "__main__":
    show_page2()