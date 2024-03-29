import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


PATH_CENTRALIZED_MODEL = "prototype_1/centralized/models/centralized-model.pth"
PATH_SCALER = "prototype_1/centralized/models/scaler_centralized_model.pkl"

PATH_TON_DATASET = "datasets/pre-processed/NF-ToN-IoT-v2.parquet"
PATH_UNSW_DATASET = "datasets/pre-processed/NF-UNSW-NB15-v2.parquet"
PATH_BOT_DATASET = "datasets/pre-processed/NF-BoT-IoT-v2.parquet"
PATH_CENTRALIZED_DATASET = "datasets/pre-processed/centralized.parquet"

TARGET_NAME = "Attack"


def get_standarlize_data(df: pd.DataFrame, path_to_scaler: str):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    scaler = joblib.load(path_to_scaler)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=69, stratify=y)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69, stratify=y_temp)

    X_eval_scaled = scaler.transform(X_eval)
    X_test_scaled = scaler.transform(X_test)

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_eval": X_eval_scaled,
        "y_eval": y_eval,
        "X_test": X_test_scaled,
        "y_test": y_test,
    }

    return data


def get_aggregate_data(df: pd.DataFrame, path_to_scaler: str):

    def get_scaler_standarlize(_X_train):
        scaler = StandardScaler()
        scaler.fit(_X_train)
        joblib.dump(scaler, path_to_scaler)

        return scaler

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=69, stratify=y)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69, stratify=y_temp)

    scaler = get_scaler_standarlize(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_eval_scaled = scaler.transform(X_eval)

    data = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_eval": X_eval_scaled,
        "y_eval": y_eval,
        "X_test": X_test_scaled,
        "y_test": y_test,
    }

    return data
