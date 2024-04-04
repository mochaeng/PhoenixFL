
import pandas as pd
import torch
import joblib

from .utils import get_standardized_data
from ..pre_process import PATH_CENTRALIZED_MODEL, TARGET_NAME, PATH_BOT_DATASET, PATH_TON_DATASET, PATH_UNSW_DATASET, PATH_SCALER
from ..neural_helper.mlp import PopoolaMLP, FnidsMLP, test_metrics, DEVICE, load_data


if __name__ == "__main__":

    datasets_paths: list[tuple[str, str]] = [
        ("client-1: ToN", PATH_TON_DATASET), 
        ("client-2: BoT", PATH_BOT_DATASET), 
        ("client-3: UNSW", PATH_UNSW_DATASET),
    ]
    
    scaler = joblib.load(PATH_SCALER)

    for dataset_name, dataset_path in datasets_paths:
        print("\nEvaluating centralized trained dataset")
        print(f"Current: {dataset_name}")

        df = pd.read_parquet(dataset_path)
        df = df.drop(columns=[TARGET_NAME])
        df = df.drop_duplicates()

        data = get_standardized_data(df, scaler)
        train_loader, eval_loader, test_loader = load_data(data)

        model = PopoolaMLP().to(device=DEVICE)

        model.load_state_dict(torch.load(PATH_CENTRALIZED_MODEL))
        model.eval()
        
        test_metrics(model, eval_loader)
        print(f'\nEvaluation: {test_metrics(model, eval_loader)}')
        print(f'Testing: {test_metrics(model, test_loader)}')
