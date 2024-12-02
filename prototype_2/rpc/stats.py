import json

import pandas as pd


def write_latencies(name, latencies):
    column = "Latency"
    latencies_pd = pd.DataFrame(latencies, columns=pd.Index([column]))
    metrics = {
        "mean": latencies_pd[column].mean(),
        "median": latencies_pd[column].median(),
        "quatile-75": latencies_pd[column].quantile(0.75),
        "quatile-95": latencies_pd[column].quantile(0.95),
        "quatile-99": latencies_pd[column].quantile(0.99),
        "max": latencies_pd[column].max(),
        "min": latencies_pd[column].min(),
    }
    path = f"data/workers/{name}"
    latencies_pd.to_csv(f"{path}_latencies.csv")
    with open(f"{path}_metrics.json", "w+") as f:
        json.dump(metrics, f, indent=4)
