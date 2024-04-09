import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict
import numpy as np
from collections import OrderedDict

from ..pre_process import read_dataset, TARGET_NAME, get_standardized_data


PATH_TON_DATASET = "datasets/pre-processed/NF-ToN-IoT-v2.parquet"
PATH_UNSW_DATASET = "datasets/pre-processed/NF-UNSW-NB15-v2.parquet"
PATH_BOT_DATASET = "datasets/pre-processed/NF-BoT-IoT-v2.parquet"

# for pycharm debugger
# PATH_TON_DATASET = "../../datasets/pre-processed/NF-ToN-IoT-v2.parquet"
# PATH_UNSW_DATASET = "../../datasets/pre-processed/NF-UNSW-NB15-v2.parquet"
# PATH_BOT_DATASET = "../../datasets/pre-processed/NF-BoT-IoT-v2.parquet"


DATASETS: list[tuple[str, str]] = [
    ("client-1: ToN", PATH_TON_DATASET),
    ("client-2: BoT", PATH_BOT_DATASET),
    ("client-3: UNSW", PATH_UNSW_DATASET),
]
# DATASETS = [PATH_TON_DATASET, PATH_BOT_DATASET, PATH_UNSW_DATASET]


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_all_federated_loaders(
    batch_size=32,
) -> dict[int, Tuple[Tuple, Tuple[DataLoader, DataLoader]]]:
    def _get_all_federated_client_data():
        all_data: Dict[int, Tuple[str, Dict]] = {}
        for idx, (dataset_name, path) in enumerate(DATASETS):
            df = read_dataset(path, TARGET_NAME)
            data_std, _ = get_standardized_data(df)
            all_data[idx] = (dataset_name, data_std)
        return all_data

    all_data = _get_all_federated_client_data()

    loaders: dict[int, Tuple[Tuple, Tuple[DataLoader, DataLoader]]] = {}
    for cid, data in all_data.items():
        name, data_std = data

        x_train_tensor = torch.tensor(data_std["X_train"], dtype=torch.float32)
        y_train_tensor = torch.tensor(data_std["y_train"], dtype=torch.float32).view(
            -1, 1
        )
        x_eval_tensor = torch.tensor(data_std["X_eval"], dtype=torch.float32)
        y_eval_tensor = torch.tensor(data_std["y_eval"], dtype=torch.float32).view(
            -1, 1
        )
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        eval_dataset = TensorDataset(x_eval_tensor, y_eval_tensor)

        if cid == 0:
            print(len(train_dataset))

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size, shuffle=False)

        loaders[cid] = ((cid, name), (train_loader, eval_loader))

    return loaders
