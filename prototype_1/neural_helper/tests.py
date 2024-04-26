import unittest
import torch
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .mlp import PopoolaMLP, DEVICE, get_train_and_test_loaders, MLP


def _load_eval_loader():
    data = _get_data()
    _, eval_loader, _ = get_train_and_test_loaders(data, batch_size=32)
    return eval_loader


def _get_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.drop(columns=["Attack"])
    df = df.drop_duplicates()

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=69, stratify=y
    )
    X_eval, X_test, y_eval, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=69, stratify=y_temp
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    X_test_scaled = scaler.transform(X_test)

    data = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_eval": X_eval_scaled,
        "y_eval": y_eval,
        "X_test": X_test_scaled,
        "y_test": y_test,
    }

    return data


class MetricsTests(unittest.TestCase):
    def __init__(
        self,
        methodName: str = "runTest",
    ) -> None:
        super().__init__(methodName)
        self.net = PopoolaMLP().to(DEVICE)
        self.eval_loader = _load_eval_loader()

    def test_accuracy(self):
        true_positives, true_negatives, positives, negatives = 0, 0, 0, 0
        accuracy = BinaryAccuracy().to(DEVICE)

        self.net.eval()

        with torch.no_grad():
            for inputs, labels in self.eval_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = self.net(inputs)
                predicted = torch.round(torch.sigmoid(outputs.data)).bool()
                labels = labels.bool()

                positives += predicted.sum().item()
                true_positives += (predicted & labels).sum().item()
                negatives += (~predicted).sum().item()
                true_negatives += (~predicted & ~labels).sum().item()

                accuracy.update(predicted.view(-1), labels.view(-1))

        equation = (true_positives + true_negatives) / (positives + negatives)
        self.assertAlmostEqual(equation, accuracy.compute().item(), places=5)

    def test_precision(self):
        true_positives, false_positives = 0, 0
        precision = BinaryPrecision().to(DEVICE)
        self.net.eval()

        with torch.no_grad():
            for inputs, labels in self.eval_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = self.net(inputs)
                predicted = torch.round(torch.sigmoid(outputs.data)).bool()
                labels = labels.bool()

                true_positives += (predicted & labels).sum().item()
                false_positives += (predicted & ~labels).sum().item()

                precision.update(predicted.view(-1), labels.view(-1))

        equation = true_positives / (true_positives + false_positives)
        self.assertAlmostEqual(equation, precision.compute().item(), places=5)

    def test_recall(self):
        true_positives, false_negatives = 0, 0
        recall = BinaryRecall().to(DEVICE)
        self.net.eval()

        with torch.no_grad():
            for inputs, labels in self.eval_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = self.net(inputs)
                predicted = torch.round(torch.sigmoid(outputs.data)).bool()
                labels = labels.bool()

                true_positives += (predicted & labels).sum().item()
                false_negatives += (~predicted & labels).sum().item()

                recall.update(predicted.view(-1), labels.view(-1))

        equation = true_positives / (true_positives + false_negatives)
        self.assertAlmostEqual(equation, recall.compute().item(), places=5)

    def test_f1_score(self):
        true_positives, false_negatives, false_positives = 0, 0, 0
        f1_score = BinaryF1Score().to(DEVICE)
        self.net.eval()

        with torch.no_grad():
            for inputs, labels in self.eval_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = self.net(inputs)
                predicted = torch.round(torch.sigmoid(outputs.data)).bool()
                labels = labels.bool()

                true_positives += (predicted & labels).sum().item()
                false_negatives += (~predicted & labels).sum().item()
                false_positives += (predicted & ~labels).sum().item()

                f1_score.update(predicted.view(-1), labels.view(-1))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        equation = (2 * precision * recall) / (precision + recall)

        self.assertAlmostEqual(equation, f1_score.compute().item(), places=5)


class MLPTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.net = MLP().to(DEVICE)
        self.single_example = _get_data()["X_train"][0]

    def test_forward_single_example(self):
        t = torch.tensor(self.single_example, dtype=torch.float32).to(DEVICE)
        print(self.net(t))


if __name__ == "__main__":
    DATA_PATH = "datasets/pre-processed/centralized.parquet"
    # DATA_PATH = "../../datasets/pre-processed/centralized.parquet"

    suite = unittest.TestSuite()
    suite.addTest(MLPTest("test_forward_single_example"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # unittest.main()
