import argparse
import json

from ..pre_process import CLIENTS_PATH, BATCH_SIZE
from ..neural_helper.mlp import train, evaluate_model, MLP, DEVICE, TRAIN_CONFIG
from .local_helpers import get_local_loaders, get_eval_test_loader

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

    global_metrics = {}
    local_train_logs_metrics = {}

    for client_name, dataset_paths in CLIENTS_PATH:
        print(f"\nTraining local DNN on {client_name} data")

        dataset_name = client_name.split(" ")[1]
        local_model_key = f"model_{dataset_name}"
        global_metrics[local_model_key] = {}

        (local_train_loader, local_test_loader), local_scaler = get_local_loaders(
            dataset_paths, BATCH_SIZE
        )

        model = MLP().to(DEVICE)
        logs = train(model, local_train_loader, TRAIN_CONFIG, is_epochs_logs=True)

        local_train_logs_metrics[client_name] = logs
        local_metrics = evaluate_model(model, local_test_loader)
        del local_metrics["final_loss"]
        global_metrics[local_model_key][client_name] = local_metrics

        for eval_client_name, eval_dataset_path in CLIENTS_PATH:
            if eval_client_name == client_name:
                continue

            print(f"Evaluating local DNN at {eval_client_name}")

            eval_test_loader = get_eval_test_loader(
                eval_dataset_path["TEST"], local_scaler, BATCH_SIZE
            )

            eval_metrics = evaluate_model(model, eval_test_loader)
            del eval_metrics["final_loss"]
            global_metrics[local_model_key][eval_client_name] = eval_metrics

            print(f"Eval {eval_client_name} has {eval_metrics}\n")

    with open(f"{PATH_TO_SAVE}/metrics.json", "w") as f:
        json.dump(global_metrics, f, indent=4)

    with open(f"{PATH_TO_SAVE}/train_logs.json", "w") as f:
        json.dump(local_train_logs_metrics, f, indent=4)
