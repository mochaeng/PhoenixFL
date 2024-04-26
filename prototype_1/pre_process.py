import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from typing import Dict, List, Tuple

PATH_CENTRALIZED_MODEL = "prototype_1/centralized/models/centralized-model.pth"
PATH_SCALER = "prototype_1/centralized/models/scaler_centralized_model.pkl"

__PREPROCESSED_TRAIN_TEST_DATASETS_PATH = "datasets/pre-processed/train-test"
# PREPROCESSED_DATASETS_PATH = "../../datasets/pre-processed"

DATASETS_PATHS = {
    "TON": {
        "TRAIN": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-1: ToN/NF-TON-IOT-V2_train.parquet",
        "TEST": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-1: ToN/NF-TON-IOT-V2_test.parquet",
    },
    "BOT": {
        "TRAIN": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-2: BoT/NF-BOT-IOT-V2_train.parquet",
        "TEST": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-2: BoT/NF-BOT-IOT-V2_test.parquet",
    },
    "UNSW": {
        "TRAIN": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-3: UNSW/NF-UNSW-NB15-V2_train.parquet",
        "TEST": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-3: UNSW/NF-UNSW-NB15-V2_test.parquet",
    },
    "CSE": {
        "TRAIN": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-4: CSE/NF-CSE-CIC-IDS2018-V2_train.parquet",
        "TEST": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-4: CSE/NF-CSE-CIC-IDS2018-V2_test.parquet",
    },
    "CENTRALIZED": {
        "TRAIN": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/centralized/centralized_train.parquet",
        "TEST": f"{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/centralized/centralized_train.parquet",
    },
}

# For pycharm:
# DATASETS_PATHS = {
#     "TON": {
#         "TRAIN": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-1: ToN/NF-TON-IOT-V2_train.parquet",
#         "TEST": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-1: ToN/NF-TON-IOT-V2_test.parquet",
#     },
#     "BOT": {
#         "TRAIN": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-2: BoT/NF-BOT-IOT-V2_train.parquet",
#         "TEST": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-2: BoT/NF-BOT-IOT-V2_test.parquet",
#     },
#     "UNSW": {
#         "TRAIN": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-3: UNSW/NF-UNSW-NB15-V2_train.parquet",
#         "TEST": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-3: UNSW/NF-UNSW-NB15-V2_test.parquet",
#     },
#     "CSE": {
#         "TRAIN": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-4: CSE/NF-CSE-CIC-IDS2018-V2_train.parquet",
#         "TEST": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/client-4: CSE/NF-CSE-CIC-IDS2018-V2_test.parquet",
#     },
#     "CENTRALIZED": {
#         "TRAIN": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/centralized/centralized_train.parquet",
#         "TEST": f"../../{__PREPROCESSED_TRAIN_TEST_DATASETS_PATH}/centralized/centralized_train.parquet",
#     },
# }

COLUMN_TO_REMOVE = "Attack"

CLIENTS_NAMES = ["client-1: ToN", "client-2: BoT", "client-3: UNSW", "client-4: CSE"]
METRICS_NAMES = ["accuracy", "precision", "recall", "f1_score"]

CLIENTS_PATH: List[Tuple[str, Dict]] = [
    ("client-1: ToN", DATASETS_PATHS["TON"]),
    ("client-2: BoT", DATASETS_PATHS["BOT"]),
    ("client-3: UNSW", DATASETS_PATHS["UNSW"]),
    ("client-4: CSE", DATASETS_PATHS["CSE"]),
]


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def get_train_test_dataframes_and_drop_column(
    path: Dict, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = path["TRAIN"], path["TEST"]
    train_df = pd.read_parquet(train_path).drop(columns=[target_column])
    test_df = pd.read_parquet(test_path).drop(columns=[target_column])
    return train_df, test_df


def get_data(df: pd.DataFrame) -> Dict:
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=69, stratify=y
    )

    # X_train, X_temp, y_train, y_temp = train_test_split(
    #     X, y, test_size=0.4, random_state=69, stratify=y
    # )
    # X_eval, X_test, y_eval, y_test = train_test_split(
    #     X_temp, y_temp, test_size=0.5, random_state=69, stratify=y_temp
    # )

    data = {
        "X_train": x_train,
        "y_train": y_train,
        "X_test": x_test,
        "y_test": y_test,
    }

    return data


def get_x_y_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    return {
        "x": x,
        "y": y,
    }


def get_standardized_data(df: pd.DataFrame, scaler=None, path_to_save_scaler=None):
    data = get_data(df)

    if scaler is None:
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaler.fit(data["X_train"])
        if path_to_save_scaler is not None:
            joblib.dump(scaler, PATH_SCALER)

    data["X_train"] = scaler.transform(data["X_train"])
    data["X_test"] = scaler.transform(data["X_test"])
    return data, scaler


def get_standarlize_client_data_from_scaler(
    df: pd.DataFrame, scaler
) -> Dict[str, np.ndarray]:
    client_data = get_x_y_data(df)
    client_data["x"] = scaler.transform(client_data["x"])
    return client_data


def get_standardized_data_from_train_test_dataframes(
    train_df: pd.DataFrame, test_df: pd.DataFrame, scaler=None, path_to_save_scaler=None
) -> Tuple[Dict[str, np.ndarray], MinMaxScaler]:
    train_data = get_x_y_data(train_df)
    test_data = get_x_y_data(test_df)
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(train_data["x"])
        if path_to_save_scaler is not None:
            joblib.dump(scaler, PATH_SCALER)

    train_data["x"] = scaler.transform(train_data["x"])
    test_data["x"] = scaler.transform(test_data["x"])

    return {
        "x_train": train_data["x"],
        "y_train": train_data["y"],
        "x_test": test_data["x"],
        "y_test": test_data["y"],
    }, scaler
