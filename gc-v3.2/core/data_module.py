import pandas as pd
import numpy as np

class DataModule:
    def __init__(self):
        self.df = None

    def load_sample_data(self):
        dates = pd.date_range("2025-01-01", periods=1000, freq="h")
        data = {
            "datetime": dates,
            "carbon_emission": np.random.normal(800, 200, 1000),
            "temperature": np.random.normal(20, 8, 1000),
            "humidity": np.random.normal(50, 15, 1000),
            "occupancy": np.random.uniform(0.2, 0.9, 1000),
            "pv_generation": np.random.uniform(0, 250, 1000),
            "hvac_power": np.random.normal(400, 100, 1000),
        }
        self.df = pd.DataFrame(data)
        return self.df

    def detect_anomalies(self, threshold=3):
        if self.df is None:
            self.load_sample_data()
        mean = self.df["carbon_emission"].mean()
        std = self.df["carbon_emission"].std()
        self.df["anomaly"] = (self.df["carbon_emission"] > mean + threshold * std)
        return self.df