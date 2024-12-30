import os

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pre_process.pre_process import COLUMNS_TO_REMOVE, get_df

DATASETS_PATH = "datasets/pre-processed/phoenix"
PATH_TO_SAVE = "datasets/data-for-prototype-02"

if __name__ == "__main__":
    ton_dataset = os.path.join(DATASETS_PATH, "NF-TON-IOT-V2.parquet")
    df = get_df(ton_dataset, is_drop=False).sample(frac=1)
    df_file_path = os.path.join(PATH_TO_SAVE, "lines.csv")
    lines = df.iloc[:10000]
    lines.to_csv(df_file_path, index=False)

    columns = [*COLUMNS_TO_REMOVE, "Label"]
    df_for_scale = df.drop(columns=columns)
    scaler = MinMaxScaler()
    scaler = scaler.fit(df_for_scale.values, df_for_scale.columns)
    scaler_file_path = os.path.join(PATH_TO_SAVE, "scaler.pkl")
    joblib.dump(scaler, scaler_file_path)

    print(lines["Attack"].value_counts())

    # print(type(df.iloc[0]))  # pandas series
    # print(len(df.iloc[0]))
    # print(len(df.iloc[0][:-1]))
    # print(df.iloc[0][:-1])
