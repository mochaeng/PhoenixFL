import json
import os

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pre_process.pre_process import get_df

DATASETS_PATH = "datasets/pre-processed/phoenix"
PATH_TO_SAVE = "datasets/data-for-prototype-02"

COLUMNS_TO_REMOVE = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
]

if __name__ == "__main__":
    ton_dataset = os.path.join(DATASETS_PATH, "NF-TON-IOT-V2.parquet")
    df = (
        get_df(ton_dataset, is_drop=False)
        .sample(frac=1)
        .drop(columns=["Label", "Attack"])
    )
    print(df)

    df_file_path = os.path.join(PATH_TO_SAVE, "lines.csv")
    lines = df.iloc[:10000]
    lines.to_csv(df_file_path, index=False)

    columns = [*COLUMNS_TO_REMOVE]
    df_for_scale = df.drop(columns=columns)
    scaler = MinMaxScaler()
    scaler = scaler.fit(df_for_scale.values, df_for_scale.columns)
    scaler_file_path = os.path.join(PATH_TO_SAVE, "scaler.pkl")
    joblib.dump(scaler, scaler_file_path)

    scaler_data = {
        "min": scaler.min_.tolist(),
        "scale": scaler.scale_.tolist(),
    }

    with open(f"{PATH_TO_SAVE}/scaler.json", "w") as f:
        json.dump(scaler_data, f)

    print(scaler_data)

    # print(lines["Attack"].value_counts())

    # print(type(df.iloc[0]))  # pandas series
    # print(len(df.iloc[0]))
    # print(len(df.iloc[0][:-1]))
    # print(df.iloc[0][:-1])
