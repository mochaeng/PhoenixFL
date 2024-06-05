import dask.dataframe as dd
import pandas as pd

# ---------------------------------------------------------
# ONLY RUN THIS CODE IF YOU HAVE MORE THAN 16GB RAM MINIMUM
#                 (YOU CAN USE KAGGLE)
# ---------------------------------------------------------

kaggle = {
    "nf-unsw-nb15-v2": {
        "path": "/kaggle/input/nf-unsw-nb15-v2/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv",
        "counts": [
            ("Analysis", 36 + 15 + 20),
            ("Backdoor", 32 + 14 + 19),
            ("Benign", 223364 + 95727 + 136660),
            ("DoS", 328 + 140 + 193),
            ("Exploits", 2884 + 1236 + 1782),
            ("Fuzzers", 1481 + 635 + 923),
            ("Generic", 268 + 115 + 193),
            ("Reconnaissance", 631 + 271 + 407),
            ("Shellcode", 57 + 24 + 46),
            ("Worms", 13 + 6 + 19),
        ],
    },
    "nf-ton-iot-v2": {
        "path": "/kaggle/input/nf-ton-iot-v2/9bafce9d380588c2_MOHANAD_A4706/data/NF-ToN-IoT-v2.csv",
        "counts": [
            ("Benign", 26500 + 11357 + 16167),
            ("backdoor", 187 + 80 + 85),
            ("ddos", 19572 + 8388 + 12040),
            ("dos", 6259 + 2682 + 3724),
            ("injection", 6726 + 2882 + 4078),
            ("mitm", 58 + 25 + 41),
            ("password", 10982 + 4706 + 6869),
            ("ransomware", 33 + 14 + 18),
            ("scanning", 10508 + 4503 + 6324),
            ("xss", 23772 + 10188 + 14692),
        ],
    },
    "nf-bot-iot-v2": {
        "path": "/kaggle/input/nf-bot-iot-vv2/befb58edf3428167_MOHANAD_A4706/data/NF-BoT-IoT-v2.csv",
        "counts": [
            ("Benign", 638 + 274 + 373),
            ("DDoS", 41383 + 17736 + 25433),
            ("DoS", 81876 + 35090 + 50038),
            ("Reconnaissance", 12687 + 5437 + 7783),
            ("Theft", 18 + 8 + 7),
        ],
    },
    "nf-cse-cic-ids2018-v2": {
        "path": "/kaggle/input/nf-cse-cic-ids2018-v2/b3427ed8ad063a09_MOHANAD_A4706/data/NF-CSE-CIC-IDS2018-v2.csv",
        "counts": [
            ("Benign", 75765 + 32471 + 46413),
            ("Bot", 750 + 321 + 439),
            ("Brute Force -Web", 8 + 4 + 12),
            ("Brute Force -XSS", 1 + 1 + 2),
            ("DDOS attack-HOIC", 5347 + 2292 + 3298),
            ("DDoS attacks-LOIC-HTTP", 1412 + 605 + 874),
            ("DoS attacks-GoldenEye", 144 + 62 + 86),
            ("DoS attacks-Hulk", 2120 + 909 + 1332),
            ("DoS attacks-SlowHTTPTest", 59 + 25 + 49),
            ("DoS attacks-Slowloris", 34 + 14 + 21),
            ("FTP-BruteForce", 135 + 58 + 74),
            ("Infilteration", 550 + 236 + 301),
            ("SQL Injection", 5 + 2 + 1),
            ("SSH-Bruteforce", 467 + 200 + 240),
        ],
    },
}

COLUMNS_TO_REMOVE = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    # "L4_DST_PORT",
]


# def get_dataset_SRS(df, distribution: list[tuple[str, int]], sample_total: int):
#     samples = []
#     for attack_name, attack_proportion in distribution:
#         pandas_df: pd.DataFrame = df[df["Attack"] == attack_name].compute()
#         # print(pandas_df.columns)
#         pandas_df = pandas_df.drop_duplicates()
#         n = round(sample_total * attack_proportion)
#         total = pandas_df["Attack"].value_counts().values[0]
#         print(f"\t{attack_name} -> {total} | {attack_proportion} | {n}")

#         if total >= n:
#             samples.append(pandas_df.sample(n=n))
#         else:
#             samples.append(pandas_df.sample(n=total))
#     return pd.concat(samples)


def get_duplicates_in_df(df: pd.DataFrame) -> list:
    df_no_attack = df.drop(columns=["Attack"])
    duplicates = df_no_attack[df_no_attack.duplicated()]
    duplicates_with_attack = pd.merge(
        duplicates, df["Attack"], left_index=True, right_index=True
    )
    result = duplicates_with_attack["Attack"].value_counts()
    return list(zip(result.index.tolist(), result.values))


# def get_stratified_random_sample(
#     df, distribution: list[tuple[str, int]]
# ) -> pd.DataFrame:
#     samples = []
#     for attack_name, values_count in distribution:
#         pandas_df: pd.DataFrame = df[df["Attack"] == attack_name].compute()
#         pandas_df = pandas_df.drop_duplicates()
#         total = pandas_df['Attack'].value_counts().values[0]
#         print(f"\t{attack_name} -> {total} | {values_count}")

#         if total >= values_count:
#             samples.append(pandas_df.sample(n=values_count))
#         else:
#             samples.append(pandas_df.sample(n=total))

#     return pd.concat(samples)


def get_dataset_SRS(df, distribution: list[tuple[str, int]]):
    result_df = pd.DataFrame(columns=df.columns)

    for attack_name, values_count in distribution:
        pandas_df: pd.DataFrame = df[df["Attack"] == attack_name].compute()
        pandas_df = pandas_df.drop_duplicates()
        desired_amount = values_count
        total_amount = pandas_df["Attack"].value_counts().values[0]
        print(f"\t{attack_name} -> {total_amount} | {desired_amount}")

        current_sample = pd.DataFrame()

        if total_amount < desired_amount:
            current_sample = pandas_df.sample(
                n=total_amount, replace=False, random_state=69
            )
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


for dataset_name in kaggle.keys():
    print(dataset_name)
    df = dd.read_csv(kaggle[dataset_name]["path"]).drop(COLUMNS_TO_REMOVE, axis=1)  # type: ignore
    stratified_sample = get_dataset_SRS(df, kaggle[dataset_name]["counts"])
    print(f"\t{get_duplicates_in_df(stratified_sample)}")

    print(stratified_sample["Attack"].value_counts())

    file_path = f"./popoola/{dataset_name.upper()}.parquet"
    print(f"\t{file_path}")
    print("\n\n")
    stratified_sample.to_parquet(file_path, compression="gzip")

print("finished")
