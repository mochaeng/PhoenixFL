import os
import argparse
import json
import pandas as pd
from scipy.stats import shapiro


METRICS_FOLDER_PATH = "federated/metrics"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analysing the results from the federated training"
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="The name of the federated algorithm you want to analyse",
        choices=["fedavg", "fedprox", "fedmedian"],
        required=True,
    )

    args = parser.parse_args()
    strategy_name = args.algo

    metrics_path = os.path.join(METRICS_FOLDER_PATH, f"metrics_{strategy_name}.json")

    if not os.path.exists(metrics_path):
        raise ValueError(
            f"no metrics found for {strategy_name}. You should train with it first for getting the results file"
        )

    with open(metrics_path, "r+") as file:
        metrics = json.load(file)

    results_path = os.path.join(METRICS_FOLDER_PATH, f"results_{strategy_name}.txt")

    del metrics["weighted_metrics"]

    with open(results_path, "w+") as file:
        for client_name in metrics.keys():
            client_metrics = metrics[client_name]

            def get_last_round_metric():
                return {
                    metric_name: list(map(lambda rounds: rounds[-1], models))
                    for metric_name, models in client_metrics.items()
                }

            values = get_last_round_metric()
            print(values)
            print()

            file.write(f"[{client_name}]\n")
            df = pd.DataFrame.from_dict(values)

            for metric in df:
                shapiro_test = shapiro(df[metric].values)
                file.write(f"\t[{metric}]:\n")
                file.write(f"\t\tmean: {df[metric].mean()}\n")
                file.write(f"\t\tmedian: {df[metric].median()}\n")
                file.write(f"\t\tstd: {df[metric].std()}\n")
                file.write(f"\t\tmax: {df[metric].max()}\n")
                file.write(f"\t\tmin: {df[metric].min()}\n")
                file.write(f"\t\tshapiro: {shapiro_test}\n")

            file.write("\n\n")
