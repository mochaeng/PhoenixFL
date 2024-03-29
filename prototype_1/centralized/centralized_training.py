import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

from ..neural_helper import mlp
from .utils import PATH_CENTRALIZED_DATASET, get_aggregate_data, PATH_SCALER, PATH_CENTRALIZED_MODEL


if __name__ == "__main__":
    df = pd.read_parquet(PATH_CENTRALIZED_DATASET)
    df = df.drop(columns=["Attack"])
    df = df.drop_duplicates()

    data = get_aggregate_data(df, PATH_SCALER)

    device = "cuda"
    model = mlp.PopoolaMLP().to(device=device)
    # model = FnidsMLP().to(device=device)

    net_helper = mlp.NetHelper(model=model, data=data, device=device, batch_size=512, lr=0.02)
    net_helper.train()

    print(f'Evaluation: {net_helper.test(is_evaluation=True)}')
    print(f'Testing: {net_helper.test(is_evaluation=False)}')

    torch.save(net_helper.model.state_dict(), PATH_CENTRALIZED_MODEL)
