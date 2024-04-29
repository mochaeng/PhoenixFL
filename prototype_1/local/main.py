import argparse
import pandas as pd
import json

from ..pre_process import (
    CLIENTS_PATH,
    get_train_and_test_dfs,
    get_standardized_data_from_train_test_dataframes,
    get_test_df,
    get_standarlize_client_data,
)
from ..neural_helper.mlp import (
    get_train_and_test_loaders,
    train,
    collect_metrics,
    MLP,
    DEVICE,
    get_test_loader,
)

PATH_TO_SAVE = "./prototype_1/local/metrics"


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
        "lr": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.0002,
    }

    global_metrics = {}

    for client_name, dataset_path in CLIENTS_PATH:
        print(f"Training local DNN on {client_name} data")

        dataset_name = client_name.split(" ")[1]
        local_model_key = f"model_{dataset_name}"
        global_metrics[local_model_key] = {}

        train_df, test_df = get_train_and_test_dfs(dataset_path)
        local_data, local_scaler = get_standardized_data_from_train_test_dataframes(
            train_df, test_df
        )

        local_train_loader, local_test_loader = get_train_and_test_loaders(
            local_data, batch_size
        )
        model = MLP().to(DEVICE)
        train(model, local_train_loader, train_config)
        local_metrics = collect_metrics(model, local_test_loader)

        global_metrics[local_model_key][client_name] = local_metrics

        for eval_client_name, eval_dataset_path in CLIENTS_PATH:
            if eval_client_name == client_name:
                continue
            print(f"Evaluating local DNN at {eval_client_name}")
            eval_test_df = get_test_df(eval_dataset_path)
            eval_data = get_standarlize_client_data(eval_test_df, local_scaler)
            eval_test_loader = get_test_loader(
                {"x_test": eval_data["x"], "y_test": eval_data["y"]}, batch_size
            )

            eval_metrics = collect_metrics(model, eval_test_loader)
            global_metrics[local_model_key][eval_client_name] = eval_metrics

            print(f"Eval {eval_client_name} has {eval_metrics}\n")

    with open(f"{PATH_TO_SAVE}/metrics.json", "w") as f:
        json.dump(global_metrics, f, indent=4)
