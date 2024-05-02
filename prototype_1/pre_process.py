import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import spmatrix
import joblib
from typing import Dict, List, Tuple, Union

PATH_CENTRALIZED_MODEL = "prototype_1/centralized/models/centralized-model.pth"
PATH_SCALER = "prototype_1/centralized/models/scaler_centralized_model.pkl"
PREPROCESSED_TRAIN_TEST_DATASETS_PATH = "datasets/pre-processed/train-test"
COLUMNS_TO_REMOVE = ["Attack"]
CLIENTS_NAMES = ["client-1: ToN", "client-2: BoT", "client-3: UNSW", "client-4: CSE"]
METRICS_NAMES = ["accuracy", "precision", "recall", "f1_score"]

DATASETS_PATHS = {
    "TON": {
        "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-1: ToN/NF-TON-IOT-V2_train.parquet",
        "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-1: ToN/NF-TON-IOT-V2_test.parquet",
    },
    "BOT": {
        "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-2: BoT/NF-BOT-IOT-V2_train.parquet",
        "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-2: BoT/NF-BOT-IOT-V2_test.parquet",
    },
    "UNSW": {
        "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-3: UNSW/NF-UNSW-NB15-V2_train.parquet",
        "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-3: UNSW/NF-UNSW-NB15-V2_test.parquet",
    },
    "CSE": {
        "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-4: CSE/NF-CSE-CIC-IDS2018-V2_train.parquet",
        "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-4: CSE/NF-CSE-CIC-IDS2018-V2_test.parquet",
    },
    "CENTRALIZED": {
        "TRAIN": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/centralized/centralized_train.parquet",
        "TEST": f"{PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/centralized/centralized_train.parquet",
    },
}
CLIENTS_PATH: List[Tuple[str, Dict]] = [
    (client_name, DATASETS_PATHS[client_name.split(" ")[1].upper()])
    for client_name in CLIENTS_NAMES
]

SCALER = MinMaxScaler
ScalerType = Union[MinMaxScaler, StandardScaler]
DataType = Union[np.ndarray, spmatrix]

BATCH_SIZE = 512


def get_df(path: str) -> pd.DataFrame:
    df = (
        pd.read_parquet(path, engine="pyarrow")
        .drop(columns=COLUMNS_TO_REMOVE)
        .drop_duplicates()
    )
    return df


def get_train_and_test_dfs(paths: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = get_df(paths["TRAIN"])
    test_df = get_df(paths["TEST"])
    return train_df, test_df


def get_x_y_data(df: pd.DataFrame) -> Dict[str, DataType]:
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    return {
        "x": x,
        "y": y,
    }


def get_standardized_data(df: pd.DataFrame, scaler: ScalerType) -> Dict[str, DataType]:
    data = get_x_y_data(df)
    data["x"] = scaler.transform(data["x"])
    return data


def get_fit_scaler_from_data(
    data: Dict[str, DataType], path_to_save=None
) -> ScalerType:
    scaler = SCALER()
    scaler.fit(data["x"])
    if path_to_save is not None:
        joblib.dump(scaler, PATH_SCALER)
    return scaler


def get_standardized_train_test_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, path_to_save=None
) -> Tuple[Tuple[Dict[str, DataType], Dict[str, DataType]], ScalerType]:
    train_data = get_x_y_data(train_df)
    test_data = get_x_y_data(test_df)

    scaler = get_fit_scaler_from_data(train_data, path_to_save)

    train_data["x"] = scaler.transform(train_data["x"])
    test_data["x"] = scaler.transform(test_data["x"])

    return (train_data, test_data), scaler


def get_prepared_data_for_loader(train_data=None, test_data=None):
    data = {}
    if train_data:
        data.update({"x_train": train_data["x"], "y_train": train_data["y"]})
    if test_data:
        data.update({"x_test": test_data["x"], "y_test": test_data["y"]})
    return data
