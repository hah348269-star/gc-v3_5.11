import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class PredictModule:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(32))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        self.model = model
        return model

    def predict_future(self, data, steps=48):
        return np.random.normal(800, 150, steps)