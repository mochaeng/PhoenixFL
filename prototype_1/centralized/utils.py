import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from ..pre_process import get_data, PATH_SCALER


def get_standardized_data(df: pd.DataFrame, scaler=None):
    data = get_data(df)
    
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data["X_train"])
        joblib.dump(scaler, PATH_SCALER)
        
    data["X_train"] = scaler.transform(data["X_train"])
    data["X_test"] = scaler.transform(data["X_test"])
    data["X_eval"] = scaler.transform(data["X_eval"])
    
    return data
    