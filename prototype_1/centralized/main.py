import argparse
import json
import os

import joblib
import torch

from centralized.centralized_helpers import (
    CentralizedMetrics,
    get_centralized_data,
    print_headers,
)
from neural.architectures import MLP
from neural.helpers import DEVICE, TRAIN_CONFIG, get_test_loader
from neural.train_eval import evaluate_model, train
from pre_process.pre_process import (
    BATCH_SIZE,
    CLIENTS_PATH,
    PATH_SCALER,
    get_df,
    get_prepared_data_for_loader,
    get_standardized_df,
)

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
        "--save-data",
        type=bool,
        help="If you want to save the models as torchscript",
        default=False,
    )

    args = parser.parse_args()
    num_models = args.num_models
    is_save_data = args.save_data

    if num_models <= 0:
        raise argparse.ArgumentTypeError("num models should be >= 1")

    train_logs = {}
    centralized_metrics = CentralizedMetrics()
    server_data = get_centralized_data(BATCH_SIZE)

    for num_model in range(num_models):
        print_headers(f"Centralized model training {num_model+1}")

        model = MLP().to(DEVICE)
        logs = train(
            net=model,
            trainloader=server_data["train_loader"],
            training_style="standard",
            train_config=TRAIN_CONFIG,
        )

        if is_save_data:
            model_scripted = torch.jit.script(model)
            torch_script_file_path = os.path.join(
                PATH_TO_MODELS, f"centralized_{num_model}_scripted.pt"
            )
            model_scripted.save(torch_script_file_path)  # type: ignore

        if logs is not None:
            train_logs[num_model] = logs

        print(f"\nEvaluating {num_model+1} model among clients")

        for client_name, dataset_path in CLIENTS_PATH:
            client_test_df = get_df(dataset_path["TEST"])
            client_test_df_scaled = get_standardized_df(
                client_test_df, server_data["scaler"]
            )

            data = get_prepared_data_for_loader(test_df=client_test_df_scaled)
            test_loader = get_test_loader(data, BATCH_SIZE)

            evaluated_metrics = evaluate_model(model, test_loader)
            centralized_metrics.add_client_name(client_name, evaluated_metrics)

            print(
                f"{client_name}:\t{centralized_metrics.get_client_metrics(client_name)}"
            )

    if is_save_data:
        scaler_file_path = os.path.join(PATH_SCALER, "centralized_scaler.pkl")
        joblib.dump(server_data["scaler"], scaler_file_path)

    print(centralized_metrics.get_metrics())

    with open(f"{PATH_TO_SAVE}/metrics.json", "w") as f:
        json.dump(centralized_metrics.get_metrics(), f, indent=4)

    with open(f"{PATH_TO_SAVE}/train_logs.json", "w") as f:
        json.dump(train_logs, f, indent=4)
