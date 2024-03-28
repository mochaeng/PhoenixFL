import pandas as pd
from mlp import NetHelper
import torch
from utils import *
from mlp import PopoolaMLP, FnidsMLP


if __name__ == "__main__":
    df = pd.read_parquet(PATH_TO_DATASET + "NF-ToN-IoT-v2.parquet")
    df = df.drop(columns=["Attack"])
    df = df.drop_duplicates()

    data = get_standarlize_data(df, PATH_TO_SCALER)

    device = "cuda"
    model = PopoolaMLP().to(device=device)
    # model = FnidsMLP().to(device=device)

    model.load_state_dict(torch.load(PATH_TO_CENTRALIZED_MODEL))
    model.eval()

    net_helper = NetHelper(model=model, data=data, device=device)

    print(f'Evaluation: {net_helper.test(is_evaluation=True)}')
    print(f'Testing: {net_helper.test(is_evaluation=False)}')
