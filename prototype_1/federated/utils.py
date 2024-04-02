import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from typing import Tuple


PATH_TON_DATASET = "datasets/pre-processed/NF-ToN-IoT-v2.parquet"
PATH_UNSW_DATASET = "datasets/pre-processed/NF-UNSW-NB15-v2.parquet"
PATH_BOT_DATASET = "datasets/pre-processed/NF-BoT-IoT-v2.parquet"

# PATH_TON_DATASET = "../../datasets/pre-processed/NF-ToN-IoT-v2.parquet"
# PATH_UNSW_DATASET = "../../datasets/pre-processed/NF-UNSW-NB15-v2.parquet"
# PATH_BOT_DATASET = "../../datasets/pre-processed/NF-BoT-IoT-v2.parquet"

DATASETS = [PATH_TON_DATASET, PATH_BOT_DATASET, PATH_UNSW_DATASET]


def _get_all_federated_client_data():
    all_data = {}
    for idx, path in enumerate(DATASETS):
        df = pd.read_parquet(path, engine="pyarrow")
        df = df.drop(columns=["Attack"])
        df = df.drop_duplicates()

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1:].values

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=69, stratify=y)
        X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69, stratify=y_temp)
        
        all_data[idx] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_eval": X_eval,
            "y_eval": y_eval,
            "X_test": X_test,
            "y_test": y_test,
        }
    return all_data       


def get_all_federated_loaders(
    batch_size=32
) -> dict[int, Tuple[DataLoader, DataLoader]]:
    
    all_data = _get_all_federated_client_data()
    loaders: dict[int, Tuple[DataLoader, DataLoader]] = {}
    for cid, data in all_data.items():
        x_train_tensor = torch.tensor(data["X_train"], dtype=torch.float32)
        y_train_tensor = torch.tensor(data["y_train"], dtype=torch.float32).view(-1, 1)
        x_eval_tensor = torch.tensor(data["X_eval"], dtype=torch.float32)
        y_eval_tensor = torch.tensor(data["y_eval"], dtype=torch.float32).view(-1, 1)
        
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        eval_dataset = TensorDataset(x_eval_tensor, y_eval_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size, shuffle=False)
        
        loaders[cid] = (train_loader, eval_loader)
    return loaders
