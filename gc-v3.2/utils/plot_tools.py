import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class PlotUtils:
    @staticmethod
    def plot_timeseries(df):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df['datetime'], df['carbon_emission'], label='碳排放')
        ax.fill_between(df.index, df['carbon_emission'].mean() + 3 * df['carbon_emission'].std(),
                        alpha=0.3, color='red')
        ax.set_title('碳排放时序')
        ax.legend()
        return fig

    @staticmethod
    def plot_feature_importance(features, importance):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(features, importance, color='#4e79a7')
        ax.set_title('特征重要性')
        return fig

    @staticmethod
    def plot_anomalies(df):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df['datetime'], df['carbon_emission'], label='正常')
        anomalies = df[df['anomaly'] == 1]
        ax.scatter(anomalies['datetime'], anomalies['carbon_emission'],
                   color='red', s=20, label='异常')
        ax.legend()
        return fig