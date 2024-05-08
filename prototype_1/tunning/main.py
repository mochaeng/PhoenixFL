import pandas as pd
import torch
import torch.optim as optim
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import GridSearchCV

from .helpers import TON_DATASET_PATH, FnidsMLP, PopoolaMLP, PATH_TO_SAVE_RESULTS
from ..pre_process import (
    COLUMNS_TO_REMOVE,
    get_standardized_data,
)


if __name__ == "__main__":
    train_df = (
        pd.read_parquet(TON_DATASET_PATH, engine="pyarrow")
        .drop(columns=[COLUMNS_TO_REMOVE])
        .sample(frac=0.1)
        .drop_duplicates()
    )
    mlp_architecture = "popoola"
    MLP = PopoolaMLP
    result = f"fnids_{str(train_df.shape)}\n"

    data = get_standardized_data(train_df)
    X = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32).view(-1, 1).squeeze()

    param_grid = {
        "batch_size": [128, 512, 1024],
        "lr": [0.1, 0.02, 0.0001],
        "optimizer": [optim.SGD, optim.Adam],
        # "optimizer__momentum": [0.0, 0.6, 0.9],
        # "weight_decay": [0.1, 0.001, 1e-4],
    }
    model = NeuralNetBinaryClassifier(MLP, verbose=False, max_epochs=10)  # type: ignore
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=3, cv=5)  # type: ignore

    grid_result = grid.fit(X, y)  # type: ignore

    # results

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    result += f"Best: {grid_result.best_score_} using {grid_result.best_params_}\n\n"

    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} ({stdev}) with: {param}")
        result += f"{mean} ({stdev}) with: {param}\n"

    with open(f"{PATH_TO_SAVE_RESULTS}/results_{mlp_architecture}.txt", "w+") as f:
        f.write(result)
