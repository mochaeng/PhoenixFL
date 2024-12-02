import os

from pre_process.pre_process import get_df

PATH_TO_SAVE = "datasets/data-for-prototype-02"

if __name__ == "__main__":
    path = "datasets/pre-processed/phoenix/NF-TON-IOT-V2.parquet"
    df = get_df(path, is_drop=False).sample(frac=1)

    file_path = os.path.join(PATH_TO_SAVE, "lines.csv")
    lines = df.iloc[:10000]
    lines.to_csv(file_path, index=False)

    print(lines["Attack"].value_counts())

    # print(type(df.iloc[0]))  # pandas series
    # print(len(df.iloc[0]))
    # print(len(df.iloc[0][:-1]))
    # print(df.iloc[0][:-1])
