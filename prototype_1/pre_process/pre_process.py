import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Tuple, Union
from result import Result, Ok, Err
import json
import os


PATH_CENTRALIZED_MODEL = "prototype_1/centralized/models/centralized-model.pth"
PATH_SCALER = "datasets/data-for-prototype-02/"

PREPROCESSED_TRAIN_TEST_DATASETS_PATH = "datasets/pre-processed/train-test"
COLUMNS_TO_REMOVE = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "Attack",
]
CLIENTS_NAMES = ["client-1: BOT", "client-2: UNSW", "client-3: CSE"]
METRICS_NAMES = ["accuracy", "precision", "recall", "f1_score"]

file_extension = "parquet"
DATASETS_PATHS = {
    # "TON": {
    #     "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-1: ToN/NF-TON-IOT-V2_train.{file_extension}",
    #     "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-1: ToN/NF-TON-IOT-V2_test.{file_extension}",
    # },
    "BOT": {
        "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-2: BoT/NF-BOT-IOT-V2_train.{file_extension}",
        "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-2: BoT/NF-BOT-IOT-V2_test.{file_extension}",
    },
    "UNSW": {
        "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-3: UNSW/NF-UNSW-NB15-V2_train.{file_extension}",
        "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-3: UNSW/NF-UNSW-NB15-V2_test.{file_extension}",
    },
    "CSE": {
        "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-4: CSE/NF-CSE-CIC-IDS2018-V2_train.{file_extension}",
        "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-4: CSE/NF-CSE-CIC-IDS2018-V2_test.{file_extension}",
    },
    "CENTRALIZED": {
        "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/centralized/centralized_train.{file_extension}",
        "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/centralized/centralized_test.{file_extension}",
    },
}
CLIENTS_PATH: List[Tuple[str, Dict]] = [
    (client_name, DATASETS_PATHS[client_name.split(" ")[1].upper()])
    for client_name in CLIENTS_NAMES
]

SCALER = MinMaxScaler
ScalerType = Union[MinMaxScaler, StandardScaler, RobustScaler]
DataType = np.ndarray

BATCH_SIZE = 1024


def get_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow").drop(columns=COLUMNS_TO_REMOVE)
    return df


def get_train_and_test_dfs(paths: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = get_df(paths["TRAIN"])
    test_df = get_df(paths["TEST"])
    return train_df, test_df


def get_standardized_df(df: pd.DataFrame, scaler: ScalerType) -> pd.DataFrame:
    x_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)  # type: ignore
    return df_scaled


def get_fit_scaler_from_df(df: pd.DataFrame, path_to_save: str):
    scaler = SCALER()
    scaler = scaler.fit(df)
    return scaler


def get_standardized_train_test_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, path_to_save=""
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], ScalerType]:
    scaler = get_fit_scaler_from_df(train_df, path_to_save)
    train_df_scaled = get_standardized_df(train_df, scaler)
    test_df_scaled = get_standardized_df(test_df, scaler)

    return (train_df_scaled, test_df_scaled), scaler


def get_prepared_data_for_loader(
    train_df: Union[pd.DataFrame, None] = None,
    test_df: Union[pd.DataFrame, None] = None,
):
    data = {}
    if train_df is not None:
        x = train_df.iloc[:, :-1].values
        y = train_df.iloc[:, -1:].values
        data.update({"x_train": x, "y_train": y})
    if test_df is not None:
        x = test_df.iloc[:, :-1].values
        y = test_df.iloc[:, -1:].values
        data.update({"x_test": x, "y_test": y})
    return data


def read_file_as_dict(folder_path: str, file_name: str) -> Result[dict, str]:
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        return Err("not such file found")

    try:
        with open(file_path) as file:
            metrics = json.load(file)
    except FileNotFoundError as err:
        print("WTF")
        return Err(f"could not read file: {err}")

    return Ok(metrics)
