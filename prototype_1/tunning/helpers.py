import os

import torch.nn as nn
import torch.nn.functional as F

PATH_TO_SAVE_RESULTS = "./prototype_1/tunning/results/"
PREPROCESSED_DATASETS_PATH = "datasets/pre-processed/datasets"
TON_DATASET_PATH = os.path.join(PREPROCESSED_DATASETS_PATH, "NF-TON-IOT-V2.parquet")
CENTRALIZED_DATASET_PATH = (
    "datasets/pre-processed/train-test/centralized/centralized_test.parquet"
)


class PopoolaMLP(nn.Module):
    def __init__(self, weight_init=nn.init.xavier_uniform_) -> None:
        super(PopoolaMLP, self).__init__()
        self.fc1 = nn.Linear(39, 128)
        self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.norm2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 1)

        weight_init(self.fc1.weight)
        weight_init(self.fc2.weight)
        weight_init(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.sigmoid(x)
        return x


class FnidsMLP(nn.Module):
    def __init__(self) -> None:
        super(FnidsMLP, self).__init__()
        self.fc1 = nn.Linear(41, 160)
        self.norm1 = nn.LayerNorm(160)
        self.fc2 = nn.Linear(160, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.sigmoid(x)  # using BCEWithLogitsLoss
        return x
