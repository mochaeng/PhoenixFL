from neural.helpers import get_train_and_test_loaders
from pre_process.pre_process import (
    DATASETS_PATHS,
    get_standardized_train_test_data,
    get_train_and_test_dfs,
    get_prepared_data_for_loader,
)


class CentralizedMetrics:
    def __init__(self) -> None:
        self.__metrics = {}

    def add_client_name(self, client_name, evaluated_metrics: dict[str, float]):
        if client_name in self.__metrics:
            for metric, value in evaluated_metrics.items():
                self.__metrics[client_name][metric].append(value)
        else:
            self.__metrics[client_name] = {
                metric: [value] for metric, value in evaluated_metrics.items()
            }

    def __repr__(self) -> str:
        return str(self.__metrics)

    def get_client_metrics(self, client_name) -> dict:
        if client_name in self.__metrics:
            return self.__metrics[client_name]
        return {}

    def get_metrics(self) -> dict:
        return self.__metrics


def get_centralized_data(batch_size: int):
    train_df, test_df = get_train_and_test_dfs(DATASETS_PATHS["CENTRALIZED"])

    (train_df_scaled, test_df_scaled), scaler = get_standardized_train_test_data(
        train_df, test_df
    )
    data = get_prepared_data_for_loader(
        train_df=train_df_scaled, test_df=test_df_scaled
    )
    train_loader, test_loader = get_train_and_test_loaders(data, batch_size)

    centralized_data = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "scaler": scaler,
    }

    return centralized_data


def print_headers(msg: str):
    print("\n" + "=" * 40)
    print(f"{msg}")
    print("Starting...")
    print("=" * 40)
