import dask.dataframe as dd
import pandas as pd
import os

# ---------------------------------------------------------
#     ONLY RUN THIS CODE IF YOU HAVE AT LEAST 30GB RAM
#            (YOU CAN USE KAGGLE, like I did)
# ---------------------------------------------------------

# change the path for the original dataset on your folder if you want to run it locally
PATH_TO_SAVE = "./phoenix"
kaggle = {
    "nf-unsw-nb15-v2": {
        "path": "/kaggle/input/nf-unsw-nb15-v2/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv",
        "frac": 0.1,
    },
    "nf-ton-iot-v2": {
        "path": "/kaggle/input/nf-ton-iot-v2/9bafce9d380588c2_MOHANAD_A4706/data/NF-ToN-IoT-v2.csv",
        "frac": 0.03,
    },
    "nf-bot-iot-v2": {
        "path": "/kaggle/input/nf-bot-iot-vv2/befb58edf3428167_MOHANAD_A4706/data/NF-BoT-IoT-v2.csv",
        "frac": 0.015,
    },
    "nf-cse-cic-ids2018-v2": {
        "path": "/kaggle/input/nf-cse-cic-ids2018-v2/b3427ed8ad063a09_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018-v2.csv",
        "frac": 0.1,
    },
}


def get_dataset_SRS(df, distribution: list[tuple[str, int]], sample_total: int):
    samples = []
    for attack_name, attack_proportion in distribution:
        pandas_df: pd.DataFrame = df[df["Attack"] == attack_name].compute()
        n = round(sample_total * attack_proportion)
        print(
            f"\t{attack_name} -> {pandas_df['Attack'].value_counts().values} | {attack_proportion} | {n}"
        )
        samples.append(pandas_df.drop_duplicates().sample(n=n))
    return pd.concat(samples)


if __name__ == "__main__":
    print("Starting processing...\n")

    for dataset_name in kaggle.keys():
        print(f"Dataset: {dataset_name}")

        df = dd.read_csv(kaggle[dataset_name]["path"])  # type: ignore
        desired_frac = kaggle[dataset_name]["frac"]

        total_records = len(df)
        sample_total = round(total_records * desired_frac)

        print(f"\tSample total: {sample_total}")

        total_count = df["Attack"].value_counts().compute()
        total_ratio = df["Attack"].value_counts(normalize=True).compute()

        proportions_table = (
            dd.concat([total_count, total_ratio], axis=1)  # type: ignore
            .sort_values(by="count")
            .compute()
        )

        distribution = list(
            zip(
                proportions_table.index.tolist(), proportions_table["proportion"].values
            )
        )

        stratified_sample = get_dataset_SRS(df, distribution, sample_total)

        file_path = f"{PATH_TO_SAVE}/{dataset_name.upper()}.parquet"
        print("\n\n")
        stratified_sample.to_parquet(file_path, compression="gzip")

    print("finished")
