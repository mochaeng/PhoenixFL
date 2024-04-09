import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict


PATH_CENTRALIZED_MODEL = "prototype_1/centralized/models/centralized-model.pth"
PATH_SCALER = "prototype_1/centralized/models/scaler_centralized_model.pkl"

PATH_TON_DATASET = "datasets/pre-processed/NF-ToN-IoT-v2.parquet"
PATH_UNSW_DATASET = "datasets/pre-processed/NF-UNSW-NB15-v2.parquet"
PATH_BOT_DATASET = "datasets/pre-processed/NF-BoT-IoT-v2.parquet"
PATH_CENTRALIZED_DATASET = "datasets/pre-processed/centralized.parquet"

TARGET_NAME = "Attack"


def read_dataset(path: str, target_column: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.drop(columns=[target_column])
    df = df.drop_duplicates()
    return df


def get_data(df: pd.DataFrame) -> Dict:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=69, stratify=y
    )
    X_eval, X_test, y_eval, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=69, stratify=y_temp
    )

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "X_test": X_test,
        "y_test": y_test,
    }

    return data


def get_standardized_data(df: pd.DataFrame, scaler=None, path_to_save_scaler=None):
    data = get_data(df)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data["X_train"])
        if path_to_save_scaler is not None:
            joblib.dump(scaler, PATH_SCALER)

    data["X_train"] = scaler.transform(data["X_train"])
    data["X_test"] = scaler.transform(data["X_test"])
    data["X_eval"] = scaler.transform(data["X_eval"])

    return data, scaler
