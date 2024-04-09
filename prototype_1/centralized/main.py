import json

from ..pre_process import (
    TARGET_NAME,
    PATH_BOT_DATASET,
    PATH_TON_DATASET,
    PATH_UNSW_DATASET,
    PATH_CENTRALIZED_DATASET,
    read_dataset,
    get_standardized_data,
)
from ..neural_helper.mlp import PopoolaMLP, DEVICE, train, collect_metrics, load_data


PATH_TO_SAVE = "./prototype_1/centralized/metrics"


def _print_headers(msg: str):
    print("\n" + "=" * 40)
    print(f"{msg}")
    print("Starting...")
    print("=" * 40)


def load_centralized_data(batch_size: int):
    df = read_dataset(PATH_CENTRALIZED_DATASET, TARGET_NAME)

    data, scaler = get_standardized_data(df)
    train_loader, eval_loader, test_loader = load_data(data, batch_size)

    server_data = {
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "test_loader": test_loader,
        "scaler": scaler,
    }

    return server_data


if __name__ == "__main__":
    batch_size = 512
    global_metrics = {}

    datasets_paths: list[tuple[str, str]] = [
        ("client-1: ToN", PATH_TON_DATASET),
        ("client-2: BoT", PATH_BOT_DATASET),
        ("client-3: UNSW", PATH_UNSW_DATASET),
    ]

    server_data = load_centralized_data(batch_size)

    num_models = 2
    for num_model in range(num_models):
        _print_headers(f"Centralized model training {num_model+1}")

        model_key = f"model_{num_model+1}"
        global_metrics[model_key] = {}
        model = PopoolaMLP().to(DEVICE)
        train(model, server_data["train_loader"], epochs=10, lr=0.002)

        print(f"\nEvaluating {num_model+1} model among clients")

        for dataset_name, dataset_path in datasets_paths:
            df = read_dataset(dataset_path, TARGET_NAME)
            data, _ = get_standardized_data(df, server_data["scaler"])

            _, _, test_loader = load_data(data, batch_size)
            metrics = collect_metrics(model, test_loader)

            global_metrics[model_key][dataset_name] = metrics

            print(f"{dataset_name}:")
            print(f"\t{metrics}:")

    json_object = json.dumps(global_metrics, indent=4)

    with open(f"{PATH_TO_SAVE}/metrics.json", "w") as f:
        f.write(json_object)
