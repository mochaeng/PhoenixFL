
import pandas as pd
import torch
from .utils import PATH_SCALER, PATH_CENTRALIZED_MODEL,  TARGET_NAME, get_standarlize_data, PATH_BOT_DATASET, PATH_TON_DATASET, PATH_UNSW_DATASET

from ..neural_helper.mlp import PopoolaMLP, FnidsMLP, test, DEVICE, load_data


if __name__ == "__main__":

    datasets_paths: list[tuple[str, str]] = [
        ("client-1: ToN", PATH_TON_DATASET), 
        ("client-2: BoT", PATH_BOT_DATASET), 
        ("client-3: UNSW", PATH_UNSW_DATASET),
    ]

    for dataset_name, dataset_path in datasets_paths:
        print("\nEvaluating centralized trained dataset")
        print(f"Current: {dataset_name}")

        df = pd.read_parquet(dataset_path)
        df = df.drop(columns=[TARGET_NAME])
        df = df.drop_duplicates()

        data = get_standarlize_data(df, PATH_SCALER)
        train_loader, eval_loader, test_loader = load_data(data)

        model = PopoolaMLP().to(device=DEVICE)

        model.load_state_dict(torch.load(PATH_CENTRALIZED_MODEL))
        model.eval()
        
        test(model, eval_loader)
        print(f'Evaluation: {test(model, eval_loader)}')
        print(f'Testing: {test(model, test_loader)}')
