from typing import Dict, Tuple
from torch.utils.data import DataLoader

from pre_process.pre_process import (
    get_train_and_test_dfs,
    get_standardized_train_test_data,
    get_prepared_data_for_loader,
    get_standardized_data,
    get_df,
    ScalerType,
)
from neural.helpers import get_train_and_test_loaders, get_test_loader


class LocalMetrics:
    def __init__(self) -> None:
        self.__metrics = {}

    def add_model_name(self, model_name):
        if model_name not in self.__metrics:
            self.__metrics[model_name] = {}

    def add_model_iteration(self, model_name):
        if model_name not in self.__metrics:
            raise ValueError(f"model_name [{model_name} not found]")

        self.__metrics[model_name] = {}

    def add_client_model_values(
        self, model_name, client_name, results: dict[str, float]
    ):
        if model_name not in self.__metrics:
            raise ValueError(f"model_name [{model_name} not found]")

        if client_name in self.__metrics[model_name]:
            for metric_name, value in results.items():
                self.__metrics[model_name][client_name][metric_name].append(value)
        else:
            self.__metrics[model_name][client_name] = {
                k: [v] for k, v in results.items()
            }

    def get_metrics(self):
        return self.__metrics


def get_local_loaders(
    paths: Dict, batch_size=32
) -> Tuple[Tuple[DataLoader, DataLoader], ScalerType]:
    train_df, test_df = get_train_and_test_dfs(paths)
    (train_data, test_data), scaler = get_standardized_train_test_data(
        train_df, test_df
    )
    data = get_prepared_data_for_loader(train_data=train_data, test_data=test_data)
    train_loader, test_loader = get_train_and_test_loaders(data, batch_size)

    return (train_loader, test_loader), scaler


def get_eval_test_loader(path: str, scaler: ScalerType, batch_size: int):
    df = get_df(path)
    test_data = get_standardized_data(df, scaler)
    data = get_prepared_data_for_loader(test_data=test_data)
    test_loader = get_test_loader(data, batch_size)
    return test_loader
