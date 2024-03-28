import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from mlp import PopoolaMLP, FnidsMLP, NetHelper
import torch


def get_data(df: pd.DataFrame):

    def get_scaler_standarlize(_X_train):
        scaler = StandardScaler()
        scaler.fit(_X_train)
        joblib.dump(scaler, './models/scaler_centralized_model.pkl')

        return scaler

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=69, stratify=y)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69, stratify=y_temp)

    scaler = get_scaler_standarlize(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_eval_scaled = scaler.transform(X_eval)

    data = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_eval": X_eval_scaled,
        "y_eval": y_eval,
        "X_test": X_test_scaled,
        "y_test": y_test,
    }

    return data


if __name__ == "__main__":
    df = pd.read_parquet("../../datasets/pre-processed/centralized.parquet")
    df = df.drop(columns=["Attack"])
    df = df.drop_duplicates()

    data = get_data(df)

    device = "cuda"

    model = PopoolaMLP().to(device=device)
    # model = FnidsMLP().to(device=device)

    net_helper = NetHelper(model=model, data=data, device=device, batch_size=512, lr=0.02)
    net_helper.train()

    print(f'Evaluation: {net_helper.test(is_evaluation=True)}')
    print(f'Testing: {net_helper.test(is_evaluation=False)}')

    torch.save(net_helper.model.state_dict(), './models/centralized-model.pth')
