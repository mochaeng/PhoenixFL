import argparse
import json

from local.local_helpers import get_local_loaders, get_eval_test_loader, LocalMetrics
from pre_process.pre_process import CLIENTS_PATH, BATCH_SIZE
from neural.helpers import DEVICE, TRAIN_CONFIG
from neural.architectures import MLP
from neural.train_eval import train, evaluate_model


PATH_TO_SAVE = "local/metrics"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulating the training of ML models")
    parser.add_argument(
        "--num-models",
        type=int,
        help="The number of models to be trained",
        default=1,
    )

    args = parser.parse_args()
    num_iterations = args.num_models

    metrics = LocalMetrics()
    global_metrics = {}
    local_train_logs_metrics = {}

    for client_name, dataset_paths in CLIENTS_PATH:
        print("-" * 50)
        print(f"\nTraining local DNN on {client_name} data")
        print("-" * 50, "\n")

        dataset_name = client_name.split(" ")[1]
        model_name = f"model_{dataset_name}"

        metrics.add_model_name(model_name)

        for num_iteration in range(1, num_iterations + 1):
            print(f"Num iteration: {num_iteration}")

            (local_train_loader, local_test_loader), local_scaler = get_local_loaders(
                dataset_paths, BATCH_SIZE
            )

            model = MLP().to(DEVICE)
            logs = train(
                net=model,
                trainloader=local_train_loader,
                training_style="standard",
                train_config=TRAIN_CONFIG,
            )

            local_train_logs_metrics[client_name] = logs
            local_metrics = evaluate_model(model, local_test_loader)
            # del local_metrics["final_loss"]

            metrics.add_client_model_values(model_name, client_name, local_metrics)

            for eval_client_name, eval_dataset_path in CLIENTS_PATH:
                if eval_client_name == client_name:
                    continue

                print(f"Evaluating local DNN at {eval_client_name}")

                eval_test_loader = get_eval_test_loader(
                    eval_dataset_path["TEST"], local_scaler, BATCH_SIZE
                )

                eval_metrics = evaluate_model(model, eval_test_loader)
                # del eval_metrics["final_loss"]

                metrics.add_client_model_values(
                    model_name, eval_client_name, eval_metrics
                )

                print(f"Eval {eval_client_name} has {eval_metrics}\n")

    with open(f"{PATH_TO_SAVE}/metrics.json", "w+") as f:
        json.dump(metrics.get_metrics(), f, indent=4)

    with open(f"{PATH_TO_SAVE}/train_logs.json", "w") as f:
        json.dump(local_train_logs_metrics, f, indent=4)
