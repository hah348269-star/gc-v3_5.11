import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

class XGBModule:
    def __init__(self):
        self.model = None
        self.feature_importance = None

    def train(self, df):
        X = df[["temperature", "humidity", "occupancy", "pv_generation", "hvac_power"]]
        y = df["carbon_emission"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
        self.model = xgb.XGBRegressor()
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_
        return self.feature_importance