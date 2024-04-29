import json
import pandas as pd
import argparse

from ..pre_process import (
    COLUMN_TO_REMOVE,
    get_standarlize_client_data,
    CLIENTS_PATH,
)
from ..neural_helper.mlp import (
    DEVICE,
    train,
    collect_metrics,
    MLP,
    get_test_loader,
)
from .centralized_helpers import print_headers, get_centralized_data


PATH_TO_SAVE = "./prototype_1/centralized/metrics"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulating the training of ML models")
    parser.add_argument(
        "--num-models",
        type=int,
        help="The number of models to be trained",
        default=1,
    )

    args = parser.parse_args()
    num_models = args.num_models

    batch_size = 512
    train_config = {
        "epochs": 10,
        "lr": 0.002,
        "momentum": 0.9,
        "weight_decay": 0.0002,
    }

    global_metrics = {}

    server_data = get_centralized_data(batch_size)

    num_models = 2
    for num_model in range(num_models):
        print_headers(f"Centralized model training {num_model+1}")

        model_key = f"model_{num_model+1}"
        global_metrics[model_key] = {}
        model = MLP().to(DEVICE)
        train(model, server_data["train_loader"], train_config)

        print(f"\nEvaluating {num_model+1} model among clients")

        for dataset_name, dataset_path in CLIENTS_PATH:
            client_df = pd.read_parquet(dataset_path["TEST"], engine="pyarrow")
            client_df = client_df.drop(columns=[COLUMN_TO_REMOVE])
            client_df = client_df.drop_duplicates()

            client_data = get_standarlize_client_data(client_df, server_data["scaler"])

            test_loader = get_test_loader(
                {"x_test": client_data["x"], "y_test": client_data["y"]}, batch_size
            )
            metrics = collect_metrics(model, test_loader)

            global_metrics[model_key][dataset_name] = metrics

            print(f"{dataset_name}:")
            print(f"\t{metrics}:")

    json_object = json.dumps(global_metrics, indent=4)
    with open(f"{PATH_TO_SAVE}/metrics.json", "w") as f:
        f.write(json_object)
