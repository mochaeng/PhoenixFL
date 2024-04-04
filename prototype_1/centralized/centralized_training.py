import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

from ..neural_helper.mlp import PopoolaMLP, FnidsMLP, train, test_metrics, load_data, DEVICE
from .utils import get_standardized_data
from ..pre_process import PATH_CENTRALIZED_DATASET, PATH_CENTRALIZED_MODEL


if __name__ == "__main__":
    df = pd.read_parquet(PATH_CENTRALIZED_DATASET)
    df = df.drop(columns=["Attack"])
    df = df.drop_duplicates()
    
    data = get_standardized_data(df)
    train_loader, eval_loader, test_loader = load_data(data, batch_size=512)
    
    model = PopoolaMLP().to(device=DEVICE)

    train(model, train_loader, epochs=10, lr=0.001)

    print(f'\nEvaluation: {test_metrics(model, eval_loader)}')
    print(f'\nTesting: {test_metrics(model, test_loader)}')

    torch.save(model.state_dict(), PATH_CENTRALIZED_MODEL)
