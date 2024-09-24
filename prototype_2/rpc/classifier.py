import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import time


class PytorchClassifier:
    def __init__(self, model_path: str, scaler_path: str) -> None:
        self.model: torch.nn.Module = torch.jit.load(model_path)
        self.scaler: MinMaxScaler = joblib.load(scaler_path)

    def get_prediction(self, data: dict):
        values_from_features = [value for _, value in data.items()]
        values = np.array([values_from_features])
        scaled_values = self.scaler.transform(values)
        tensor_values = torch.tensor(scaled_values, dtype=torch.float32).to("cuda")
        self.model.eval()
        output = self.model(tensor_values)
        prediction = torch.round(torch.sigmoid(output))
        return prediction.item()
