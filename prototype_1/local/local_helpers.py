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
from neural_helper.mlp import get_train_and_test_loaders, get_test_loader


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
