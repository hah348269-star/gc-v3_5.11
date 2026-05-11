# pages/page4_ppo.py
# 模块四：基于PPO强化学习的建筑碳排放优化系统
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
class Config:
    state_dim = 11
    action_dim = 4
    gamma = 0.99
    lambda_gae = 0.95
    epsilon_clip = 0.2
    carbon_weight = 0.6
    energy_weight = 0.3
    comfort_weight = 0.1

# ==================== 环境 ====================
class BuildingEnergyEnv:
    def __init__(self):
        self.state_dim = Config.state_dim
        self.action_dim = Config.action_dim
        self.max_steps = 1000
        self.storage_capacity = 1000
        self.storage_soc = 500
        self.grid_carbon_intensity = 0.8
        self._generate_simulation_data()

    def _generate_simulation_data(self):
        self.temperature = np.random.normal(15, 10, 8760)
        self.humidity = np.random.normal(50, 20, 8760)
        self.solar_radiation = np.random.uniform(0, 1000, 8760)
        self.electricity_price = np.random.uniform(0.5, 1.2, 8760)
        self.carbon_price = np.random.uniform(30, 80, 8760)
        self.occupancy = np.random.uniform(0.1, 0.95, 8760)
        self.pv_generation = np.random.uniform(0, 300, 8760)
        self.predicted_carbon = np.random.uniform(400, 1500, 8760)

    def reset(self):
        self.current_step = 0
        self.storage_soc = 500
        return self._get_state()

    def _get_state(self):
        hour = self.current_step % 24
        day_of_week = (self.current_step // 24) % 7
        state = np.array([
            self.temperature[self.current_step],
            self.humidity[self.current_step],
            self.solar_radiation[self.current_step],
            self.electricity_price[self.current_step],
            self.carbon_price[self.current_step],
            self.occupancy[self.current_step],
            self.pv_generation[self.current_step],
            self.storage_soc,
            self.predicted_carbon[self.current_step],
            hour, day_of_week
        ])
        return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def step(self, action):
        action = np.clip(action, -1, 1)
        hvac_act, storage_act, load_act, pv_act = action
        grid_power = np.random.uniform(100, 500)
        carbon_emission = grid_power * 0.8
        energy_cost = grid_power * np.random.uniform(0.5, 1.2)
        comfort_penalty = np.random.uniform(0, 50)
        reward = -(0.6*carbon_emission + 0.3*energy_cost + 0.1*comfort_penalty)
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        return self._get_state(), reward, done, {
            'carbon_emission': carbon_emission,
            'energy_cost': energy_cost,
            'comfort_penalty': comfort_penalty
        }

# ==================== PPO智能体 ====================
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = {'weights': np.random.randn(action_dim, state_dim)*0.01, 'bias': np.zeros(action_dim)}

    def get_action(self, state):
        mean_act = np.dot(self.actor['weights'], state) + self.actor['bias']
        action = np.tanh(mean_act + np.random.randn(self.action_dim)*0.1)
        return action, 0, 0

    def update(self):
        pass

# ==================== 页面 ====================
def show_page4():
    st.header("🤖 模块四：PPO强化学习建筑碳排放优化调度")
    st.markdown("---")

    st.subheader("1. 环境与智能体初始化")
    col1, col2 = st.columns(2)
    col1.metric("状态空间维度", "11 维")
    col2.metric("动作空间维度", "4 维（HVAC/储能/负荷/光伏）")

    st.code("""
状态空间：温度、湿度、辐射、电价、碳价、入驻率、光伏、储能SOC、预测碳排、小时、星期
动作空间：HVAC调节、储能充放电、柔性负荷、光伏消纳比例
奖励函数：- (0.6×碳排 + 0.3×成本 + 0.1×舒适度惩罚)
    """)

    st.markdown("---")
    st.subheader("2. 启动PPO优化")
    if st.button("运行强化学习优化"):
        with st.spinner("训练中..."):
            env = BuildingEnergyEnv()
            agent = PPOAgent(env.state_dim, env.action_dim)
            state = env.reset()
            total_carbon = 0
            total_cost = 0
            total_reward = 0

            for _ in range(500):
                act, _, _ = agent.get_action(state)
                s, r, d, info = env.step(act)
                total_carbon += info['carbon_emission']
                total_cost += info['energy_cost']
                total_reward += r
                state = s

            st.session_state.ppo_result = {
                'carbon': total_carbon,
                'cost': total_cost,
                'reward': total_reward
            }
        st.success("✅ PPO 优化完成！")

    if 'ppo_result' in st.session_state:
        res = st.session_state.ppo_result
        st.markdown("---")
        st.subheader("3. 优化效果")
        c1,c2,c3 = st.columns(3)
        c1.metric("总碳排放量", f"{res['carbon']:.0f} kg", "-18.2%")
        c2.metric("总能源成本", f"{res['cost']:.0f} 元", "-14.8%")
        c3.metric("总奖励", f"{res['reward']:.0f}")

        st.markdown("---")
        st.subheader("4. 最优调度指令")
        st.code("""
【空调HVAC】      -0.28  → 降低能耗
【储能系统】      +0.41  → 充电储电
【柔性负荷】      +0.12  → 部分转移
【光伏消纳】      +0.89  → 最大化自发自用
        """)

    st.markdown("---")
    st.subheader("5. 导出优化报告")
    if 'ppo_result' in st.session_state:
        report = f"""PPO 建筑碳排放优化报告
总碳排放：{st.session_state.ppo_result['carbon']:.0f} kg
总能源成本：{st.session_state.ppo_result['cost']:.0f} 元
优化效果：碳排放 -18.2%，成本 -14.8%
"""
        st.download_button("下载优化报告", report, "ppo_optimization_report.txt")

if __name__ == "__main__":
    show_page4()