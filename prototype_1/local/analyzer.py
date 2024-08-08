import os
import json
import pandas as pd
from scipy.stats import shapiro

METRICS_PATH = "local/metrics"
FILE_NAME = "metrics.json"
RESULTS_PATH = "local/metrics"

if __name__ == "__main__":
    file_path = os.path.join(METRICS_PATH, FILE_NAME)

    with open(file_path, "r+") as file:
        metrics: dict = json.load(file)

    results_path = os.path.join(METRICS_PATH, "results.txt")

    file = open(results_path, "w+")

    # group_by = {}
    for model_name in metrics.keys():
        # group_by[model_name] = {}

        file.write(f"[{model_name}]\n")

        for client_name in metrics[model_name].keys():
            values = metrics[model_name][client_name]
            # group_by[model_name][client_name] = pd.DataFrame.from_dict(values)
            df = pd.DataFrame.from_dict(values)

            # stats (mean, median, std, max, min, shapiro-wilk)
            def stats_for_metric(df, f):
                for metric in df:
                    shapiro_test = shapiro(df[metric].values)
                    f.write(f"\t\t{metric}: ")

                    data = {
                        "mean": float(df[metric].mean()),
                        "median": float(df[metric].median()),
                        "std": float(df[metric].std()),
                        "max": float(df[metric].max()),
                        "min": float(df[metric].min()),
                        "shapiro": shapiro_test,
                    }
                    f.write(str(data) + "\n")

            file.write(f"\t[{client_name}]\n")

            stats_for_metric(df, file)
            file.write("\n")

    file.close()
