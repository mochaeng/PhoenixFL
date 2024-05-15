import argparse
import json
import joblib
import os
import torch

from pre_process.pre_process import (
    CLIENTS_PATH,
    get_df,
    get_standardized_data,
    get_prepared_data_for_loader,
    BATCH_SIZE,
    PATH_SCALER,
)
from neural_helper.mlp import (
    DEVICE,
    train,
    evaluate_model,
    MLP,
    get_test_loader,
    TRAIN_CONFIG,
)
from centralized.centralized_helpers import print_headers, get_centralized_data


PATH_TO_SAVE = "centralized/metrics"
PATH_TO_MODELS = "datasets/data-for-prototype-02"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulating the training of ML models")
    parser.add_argument(
        "--num-models",
        type=int,
        help="The number of models to be trained",
        default=1,
    )
    parser.add_argument(
        "--save-scaler",
        type=bool,
        help="If you want to save the min-max scaler as an object file",
        default=True,
    )
    parser.add_argument(
        "--save-models",
        type=bool,
        help="If you want to save the models as torchscript",
        default=True,
    )

    args = parser.parse_args()
    num_models = args.num_models
    is_save_scaler = args.save_scaler
    is_save_models = args.save_models

    if num_models <= 0:
        raise argparse.ArgumentTypeError("num models should be >= 1")

    train_logs = {}
    global_metrics = {}
    server_data = get_centralized_data(BATCH_SIZE)
    num_models = num_models

    for num_model in range(num_models):
        print_headers(f"Centralized model training {num_model+1}")

        model_key = f"model_{num_model+1}"
        global_metrics[model_key] = {}

        model = MLP().to(DEVICE)
        logs = train(
            model, server_data["train_loader"], TRAIN_CONFIG, is_epochs_logs=True
        )
        if is_save_models:
            model_scripted = torch.jit.script(model)
            torch_script_file_path = os.path.join(
                PATH_TO_MODELS, f"centralized_{num_model}_scripted.pt"
            )
            model_scripted.save(torch_script_file_path)
        train_logs[num_model] = logs

        print(f"\nEvaluating {num_model+1} model among clients")

        for dataset_name, dataset_path in CLIENTS_PATH:
            client_test_df = get_df(dataset_path["TEST"])
            client_test_data = get_standardized_data(
                client_test_df, server_data["scaler"]
            )

            data = get_prepared_data_for_loader(test_data=client_test_data)
            test_loader = get_test_loader(data, BATCH_SIZE)

            metrics = evaluate_model(model, test_loader)
            global_metrics[model_key][dataset_name] = metrics

            print(f"{dataset_name}:\t{metrics}")

    if is_save_scaler:
        scaler_file_path = os.path.join(PATH_SCALER, "centralized_scaler.pkl")
        joblib.dump(server_data["scaler"], scaler_file_path)

    with open(f"{PATH_TO_SAVE}/metrics.json", "w") as f:
        json.dump(global_metrics, f, indent=4)
    with open(f"{PATH_TO_SAVE}/train_logs.json", "w") as f:
        json.dump(train_logs, f, indent=4)
