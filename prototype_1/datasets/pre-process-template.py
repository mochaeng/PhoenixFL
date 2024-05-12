import dask.dataframe as dd
import pandas as pd
from typing import List, Tuple


kaggle = {
    "nf-unsw-nb15-v2": "/kaggle/input/nf-unsw-nb15-v2/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv",
    "nf-ton-iot-v2": "/kaggle/input/nf-ton-iot-v2/9bafce9d380588c2_MOHANAD_A4706/data/NF-ToN-IoT-v2.csv",
}

local = {
    "nf-unsw-nb15-v2": "nf-unsw-nb15-v2/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv",
    "nf-ton-iot-v2": "nf-ton-iot-v2/9bafce9d380588c2_MOHANAD_A4706/data/NF-ToN-IoT-v2.csv",
}

columns_to_drop = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"


def get_stratified_sample(df: pd.DataFrame, distribution: List[Tuple[str, int]]):
    attack_group = df.groupby("Attack")
    samples = []
    for attack_name, desired_total in distribution:
        samples.append(
            attack_group.get_group(attack_name)
            .drop_duplicates()
            .sample(n=desired_total)
        )
    stratified_sample = pd.concat(samples)
    return stratified_sample


if __name__ == "__main__":
    current_dataset_name = "nf-unsw-nb15-v2"
    df = pd.read_csv(local[current_dataset_name])

    total_count = df["Attack"].value_counts()
    total_ratio = df["Attack"].value_counts(normalize=True).round(4) * 100
    expected_sample_count = (
        df.groupby("Attack")
        .apply(
            lambda group: group.sample(frac=0.15),
            include_groups=True,
        )
        .droplevel(0)["Attack"]
        .value_counts()
    )

    table_1 = pd.concat([total_count, total_ratio, expected_sample_count], axis=1)
    attacks_names = table_1.index.tolist()
    distribution = list(zip(attacks_names, expected_sample_count.values))

    df = df.drop(columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])

    stratified_sample = get_stratified_sample(df, distribution)
