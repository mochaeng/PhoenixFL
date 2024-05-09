import pandas as pd
import torch
import torch.optim as optim
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import GridSearchCV
import os

from .helpers import (
    TON_DATASET_PATH,
    FnidsMLP,
    PopoolaMLP,
    PATH_TO_SAVE_RESULTS,
    CENTRALIZED_DATASET_PATH,
)
from ..pre_process import (
    COLUMNS_TO_REMOVE,
    get_standardized_data,
    get_df,
    get_fit_scaler_from_df,
)


if __name__ == "__main__":
    mlp_architecture = "popoola"
    MLP = PopoolaMLP

    train_df = get_df(CENTRALIZED_DATASET_PATH).sample(frac=0.1)
    scaler = get_fit_scaler_from_df(train_df)
    data = get_standardized_data(train_df, scaler)

    X = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32).view(-1, 1).squeeze()

    result = f"fnids_{str(train_df.shape)}\n"

    param_grid = {
        "batch_size": [64, 128, 512, 1024],
        "lr": [0.1, 0.02, 0.0001],
        "optimizer": [optim.SGD],
        "optimizer__momentum": [0.0, 0.6, 0.9],
        "optimizer__weight_decay": [0.01, 0.001, 0.0001],
    }
    model = NeuralNetBinaryClassifier(MLP, verbose=False, max_epochs=10)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=3, cv=5)  # type: ignore

    grid_result = grid.fit(X, y)  # type: ignore

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    result += f"Best: {grid_result.best_score_} using {grid_result.best_params_}\n\n"

    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} ({stdev}) with: {param}")
        result += f"{mean} ({stdev}) with: {param}\n"

    path = os.path.join(
        PATH_TO_SAVE_RESULTS, f"results_momentum_{mlp_architecture}.txt"
    )
    with open(path, "w+") as f:
        f.write(result)
