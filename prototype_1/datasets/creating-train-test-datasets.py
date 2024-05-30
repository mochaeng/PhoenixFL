import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse


def get_df(x, y, columns, distribution_name: str):
    df = pd.DataFrame(data=x, columns=columns)
    df["Attack"] = np.squeeze(y)
    train_distribution = (
        f"{distribution_name}\n{df['Attack'].value_counts().to_string()}"
    )
    return df, train_distribution


def get_paths(file_extension):
    return [
        ("client-1: ToN", f"NF-TON-IOT-V2.{file_extension}"),
        ("client-2: BoT", f"NF-BOT-IOT-V2.{file_extension}"),
        ("client-3: UNSW", f"NF-UNSW-NB15-V2.{file_extension}"),
        ("client-4: CSE", f"NF-CSE-CIC-IDS2018-V2.{file_extension}"),
    ]


PATH_TO_PREPROCESSED = "datasets/pre-processed"
TRAIN_TEST_PATH = f"{PATH_TO_PREPROCESSED}/train-test"
PREPROCESSED_PATHS = {
    "phoenix": f"{PATH_TO_PREPROCESSED}/phoenix",
    "std": f"{PATH_TO_PREPROCESSED}/std",
    "popoola": f"{PATH_TO_PREPROCESSED}/popoola",
    "ita": f"{PATH_TO_PREPROCESSED}/ita",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creating train and test datasets by client"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Which pre-processed datasets to be used",
        choices=["phoenix", "std", "popoola", "ita"],
    )
    parser.add_argument(
        "--test-perc",
        type=float,
        help="The percentual of test distribution",
        default=0.2,
    )

    args = parser.parse_args()
    test_size = args.test_perc
    processed_name = args.name

    if test_size < 0.1 or test_size > 0.5:
        raise argparse.ArgumentTypeError(
            "Value of test size must be between 0.1 and 0.5"
        )

    datasets_path = PREPROCESSED_PATHS.get(processed_name)
    if datasets_path is None:
        raise ValueError(f"no preproccesed called {processed_name}")

    match processed_name:
        case "ita":
            file_extension = "csv"
        case _:
            file_extension = "parquet"

    clients_path = get_paths(file_extension)

    print("Starting...\n")

    for client_name, file_name in clients_path:
        print(f"Creating dataset for: {client_name}")
        client_file_path = os.path.join(datasets_path, file_name)

        match file_extension:
            case "csv":
                df = pd.read_csv(client_file_path)
            case "parquet":
                df = pd.read_parquet(client_file_path, engine="pyarrow")

        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1:].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=69, stratify=y, shuffle=True
        )

        columns = df.columns.delete(-1)

        df_train, train_distribution = get_df(x_train, y_train, columns, "train")
        df_test, test_distribution = get_df(x_test, y_test, columns, "test")

        name = file_name.split(".")[0]
        save_path = os.path.join(TRAIN_TEST_PATH, client_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(f"{save_path}/train_dist.txt", "w") as f:
            f.write(train_distribution)

        with open(f"{save_path}/test_dist.txt", "w") as f:
            f.write(test_distribution)

        df_train.to_parquet(f"{save_path}/{name}_train.parquet", compression="gzip")
        df_test.to_parquet(f"{save_path}/{name}_test.parquet", compression="gzip")

    print("Creating centralized dataset")

    train_dfs = []
    test_dfs = []
    for client_name, file_name in clients_path:
        name = file_name.split(".")[0]
        train_file_path = f"{TRAIN_TEST_PATH}/{client_name}/{name}_train.parquet"
        test_file_path = f"{TRAIN_TEST_PATH}/{client_name}/{name}_test.parquet"

        df_train = pd.read_parquet(train_file_path, engine="pyarrow")
        df_test = pd.read_parquet(test_file_path, engine="pyarrow")

        train_dfs.append(df_train)
        test_dfs.append(df_test)

    train_centralized = pd.concat(train_dfs)
    test_centralized = pd.concat(test_dfs)

    path_to_centralized = os.path.join(TRAIN_TEST_PATH, "centralized")
    if not os.path.exists(path_to_centralized):
        os.makedirs(path_to_centralized)

    train_centralized.to_parquet(f"{path_to_centralized}/centralized_train.parquet")
    test_centralized.to_parquet(f"{path_to_centralized}/centralized_test.parquet")

    train_centralized_distribution = (
        f"train\n{train_centralized['Attack'].value_counts().to_string()}"
    )
    test_centralized_distribution = (
        f"test\n{test_centralized['Attack'].value_counts().to_string()}"
    )

    with open(f"{path_to_centralized}/train_dist.txt", "w") as f:
        f.write(train_centralized_distribution)

    with open(f"{path_to_centralized}/test_dist.txt", "w") as f:
        f.write(test_centralized_distribution)
