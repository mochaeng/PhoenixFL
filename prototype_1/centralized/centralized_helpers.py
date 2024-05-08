from typing import List, Dict
import pandas as pd

from ..neural_helper.mlp import get_train_and_test_loaders
from ..pre_process import (
    DATASETS_PATHS,
    get_standardized_train_test_data,
    get_train_and_test_dfs,
    get_prepared_data_for_loader,
)


def get_centralized_data(batch_size: int):
    train_df, test_df = get_train_and_test_dfs(DATASETS_PATHS["CENTRALIZED"])

    (train_data, test_data), scaler = get_standardized_train_test_data(
        train_df, test_df
    )

    data = get_prepared_data_for_loader(train_data=train_data, test_data=test_data)
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
