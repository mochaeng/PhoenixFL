import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


PATH_TO_CENTRALIZED_MODEL = "./models/centralized-model.pth"
PATH_TO_SCALER = "./models/scaler_centralized_model.pkl"
PATH_TO_DATASET = "../../datasets/pre-processed/"


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
