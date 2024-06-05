import dask.dataframe as dd
import pandas as pd


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
        "frac": 0.006,
    },
    "nf-cse-cic-ids2018-v2": {
        "path": "/kaggle/input/nf-cse-cic-ids2018-v2/b3427ed8ad063a09_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018-v2.csv",
        "frac": 0.02,
    },
}

COLUMNS_TO_REMOVE = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    # "L4_DST_PORT",
]


def get_duplicates_in_df(df: pd.DataFrame):
    df_no_attack = df.drop(columns=["Attack"])
    duplicates = df_no_attack[df_no_attack.duplicated()]
    duplicates_with_attack = pd.merge(
        duplicates, df["Attack"], left_index=True, right_index=True
    )
    result = duplicates_with_attack["Attack"].value_counts()
    return result


def get_dataset_SRS(df, distribution: list[tuple[str, int]], sample_total: int):
    #     pandas_df: pd.DataFrame = df.compute()
    #     return pandas_df.sample(frac=0.1, replace=False, random_state=79)

    result_df = pd.DataFrame(columns=df.columns)

    for attack_name, attack_proportion in distribution:
        pandas_df: pd.DataFrame = df[df["Attack"] == attack_name].compute()
        pandas_df = pandas_df.drop_duplicates()
        desired_amount = round(sample_total * attack_proportion)
        total_amount = pandas_df["Attack"].value_counts().values[0]

        print(
            f"\t{attack_name} -> {total_amount} | {attack_proportion} | {desired_amount}"
        )

        current_sample = pd.DataFrame()
        if total_amount < desired_amount:
            current_sample = pandas_df.sample(
                n=total_amount, replace=False, random_state=69
            )
            break
        else:
            needed = desired_amount

            while needed > 0:
                sample_df = pandas_df.sample(
                    n=min(needed, total_amount), replace=False, random_state=69
                )
                sample_df_without_attack = sample_df.drop(columns=["Attack"])
                result_df_without_attack = result_df.drop(columns=["Attack"])

                is_duplicate = sample_df_without_attack.apply(tuple, axis=1).isin(
                    result_df_without_attack.apply(tuple, axis=1)
                )
                non_duplicate = sample_df[~is_duplicate]

                current_sample = pd.concat([current_sample, non_duplicate])
                needed -= len(non_duplicate)

                if len(non_duplicate) == 0:
                    break

                pandas_df = pandas_df[~pandas_df.index.isin(non_duplicate.index)]
                total_amount = len(pandas_df)

        result_df = pd.concat([result_df, current_sample])

    return result_df


if __name__ == "__main__":
    print("Starting processing...\n")

    for dataset_name in kaggle.keys():
        print(f"Dataset: {dataset_name}")

        df = (
            dd.read_csv(kaggle[dataset_name]["path"])  # type: ignore
            .drop(COLUMNS_TO_REMOVE, axis=1)
            .drop_duplicates()
        )
        desired_frac = kaggle[dataset_name]["frac"]

        total_records = len(df)
        sample_total = round(total_records * desired_frac)

        print(
            f"\tSample total: {sample_total} from {total_records} with {desired_frac}"
        )

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
        print(f"\n\t{get_duplicates_in_df(stratified_sample)}")

        file_path = f"./phoenix/{dataset_name.upper()}.parquet"
        print("\n\n")
        stratified_sample.to_parquet(file_path, compression="gzip")

    print("finished")
