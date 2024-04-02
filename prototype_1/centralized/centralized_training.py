import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

from ..neural_helper.mlp import PopoolaMLP, FnidsMLP, train, test, load_data, DEVICE
from .utils import PATH_CENTRALIZED_DATASET, get_aggregate_data, PATH_SCALER, PATH_CENTRALIZED_MODEL


if __name__ == "__main__":
    df = pd.read_parquet(PATH_CENTRALIZED_DATASET)
    df = df.drop(columns=["Attack"])
    df = df.drop_duplicates()

    data = get_aggregate_data(df, PATH_SCALER)
    train_loader, eval_loader, test_loader = load_data(data, batch_size=512)
    
    model = PopoolaMLP().to(device=DEVICE)

    train(model, train_loader, epochs=10, lr=0.02)
    # net_helper = DetectorModel(model=model, data=data, device=device, lr=0.02)
    # net_helper.train(is_verbose=True)

    print(f'Evaluation: {test(model, eval_loader)}')
    print(f'Testing: {test(model, test_loader)}')

    torch.save(model.state_dict(), PATH_CENTRALIZED_MODEL)
