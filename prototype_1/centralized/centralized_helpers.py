from typing import List, Dict
import pandas as pd

from ..neural_helper.mlp import get_train_and_test_loaders
from ..pre_process import (
    DATASETS_PATHS,
    read_dataset,
    get_standardized_data_from_train_test_dataframes,
    get_train_test_dataframes_and_drop_column,
    COLUMN_TO_REMOVE,
    get_data,
)


class GroupByCentralizedMetricsRecorder:
    def __init__(
        self,
        json_dict: Dict,
        client_names: List[str],
        metrics_names: List[str],
    ) -> None:
        self.client_names = client_names
        self.metrics_names = metrics_names
        self.__dict = {
            client_name: {metric_name: [] for metric_name in self.metrics_names}
            for client_name in self.client_names
        }
        self.__metrics = self.__groupby_metrics_into_dict(json_dict, self.__dict)
        self.__metrics_as_df = self.__get_metrics_as_dataframe(self.__metrics)

    def __groupby_metrics_into_dict(
        self, all_metrics: dict, metrics_dict: dict
    ) -> dict:
        for _, client in all_metrics.items():
            for client_name, client_metrics in client.items():
                for metric_name, value in client_metrics.items():
                    metrics_dict[client_name][metric_name].append(value)
        return metrics_dict

    def __get_metrics_as_dataframe(self, metrics_dict: dict) -> dict[str, pd.DataFrame]:
        metrics_df = {
            client_name: pd.DataFrame.from_dict(metrics_dict[client_name])
            for client_name in metrics_dict.keys()
        }
        return metrics_df

    def as_df(self):
        return self.__metrics_as_df

    def as_dict(self):
        return {name: dataframe.to_dict() for name, dataframe in self.as_df().items()}


def get_centralized_data(batch_size: int):
    train_df, test_df = get_train_test_dataframes_and_drop_column(
        DATASETS_PATHS["CENTRALIZED"], COLUMN_TO_REMOVE
    )

    data, scaler = get_standardized_data_from_train_test_dataframes(train_df, test_df)
    # data, scaler = get_standardized_data_from_values(data)
    train_loader, test_loader = get_train_and_test_loaders(data, batch_size)

    centralized_data = {
        "train_loader": train_loader,
        # "eval_loader": eval_loader,
        "test_loader": test_loader,
        "scaler": scaler,
    }

    return centralized_data


def print_headers(msg: str):
    print("\n" + "=" * 40)
    print(f"{msg}")
    print("Starting...")
    print("=" * 40)
