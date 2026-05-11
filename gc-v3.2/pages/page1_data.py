import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def show_page1():
    st.title("📊 数据展示与碳排放可视化")
    st.markdown("---")

    st.subheader("1️⃣ 上传数据文件（Excel/CSV）")
    uploaded_file = st.file_uploader("上传你的碳排放表格", type=["xlsx", "csv"])

    df = None

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.session_state.df = df
        st.success("✅ 上传成功！")

    if df is None:
        dates = pd.date_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            "date": dates.tolist() * 3,
            "building_name": ["建筑A"]*10 + ["建筑B"]*10 + ["建筑C"]*10,
            "daily_emission": np.random.randint(200, 600, 30)
        })

    st.markdown("---")
    st.subheader("2️⃣ 数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("3️⃣ 碳排放趋势图")

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in df["building_name"].unique():
        sub = df[df["building_name"] == name]
        ax.plot(sub["date"], sub["daily_emission"], marker="o", label=name)

    ax.set_title("各建筑日碳排放趋势")
    ax.set_xlabel("日期")
    ax.set_ylabel("碳排放量 (kg)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

if __name__ == "__main__":
    show_page1()