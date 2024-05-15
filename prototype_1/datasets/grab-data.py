import os

from pre_process.pre_process import CLIENTS_PATH, get_df

PATH_TO_SAVE = "datasets/data-for-prototype-02"

if __name__ == "__main__":
    dataset_name, paths = CLIENTS_PATH[0]
    df = get_df(paths["TEST"]).sample(frac=1)

    file_path = os.path.join(PATH_TO_SAVE, "lines.csv")
    lines = df.iloc[:100]
    lines.to_csv(file_path)

    print(lines)

    # first_line.to_json(file_path, indent=4)

    # print(type(df.iloc[0]))  # pandas series
    # print(len(df.iloc[0]))
    # print(len(df.iloc[0][:-1]))
    # print(df.iloc[0][:-1])
