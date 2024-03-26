import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from mlp import MLP, NetHelper

if __name__ == "__main__":
    df = pd.read_parquet("../datasets/pre-processed/centralized.parquet")
    df = df.drop(columns=["Attack"])
    df = df.drop_duplicates()

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=69, stratify=y)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69, stratify=y_temp)

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "X_test": X_test,
        "y_test": y_test,
    }

    del X, y, df

    device = "cuda"

    model = MLP().to(device=device)
    net_helper = NetHelper(model=model, data=data, device=device, batch_size=512, lr=0.001)

    net_helper.train()

    print(f'Evaluation: {net_helper.test(is_evaluation=True)}')
    print(f'Testing: {net_helper.test(is_evaluation=False)}')
