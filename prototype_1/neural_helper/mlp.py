import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class PopoolaMLP(nn.Module):
    def __init__(self) -> None:
        super(PopoolaMLP, self).__init__()
        self.fc1 = nn.Linear(41, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.sigmoid(x)  # using BCEWithLogitsLoss 
        return x
    

class FnidsMLP(nn.Module):
    def __init__(self) -> None:
        super(FnidsMLP, self).__init__()
        self.fc1 = nn.Linear(41, 160)
        self.fc2 = nn.Linear(160, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.sigmoid(x)  # using BCEWithLogitsLoss 
        return x

    
class NetHelper:

    def __init__(self, model: nn.Module, data: dict, lr=0.0001, batch_size=32, epochs=10, device="cuda") -> None:
        self.lr = lr
        self.epochs = epochs
        self.model: nn.Module = model
        self.batch_size = batch_size
        self.device = device

        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_loader, self.eval_loader, self.test_loader = self.load_data(data)

    
    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            total_loos = 0.0
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loos += loss.item()

            avg_loss = total_loos / len(self.train_loader)
            print(f'Epoch [{epoch+1}], AVG: {avg_loss}')

        return loss.data

    def test(self, is_evaluation: bool):
        if is_evaluation:
            test_loader = self.eval_loader
        else:
            test_loader = self.test_loader
        
        self.model.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(inputs)
                # predicted = (torch.sigmoid(outputs) > 0.5).float()
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accurcy = correct / total
        return accurcy

    def load_data(self, data: dict):
        x_train_tensor = torch.tensor(data["X_train"], dtype=torch.float32)
        y_train_tensor = torch.tensor(data["y_train"], dtype=torch.float32).view(-1, 1)
        x_eval_tensor = torch.tensor(data["X_eval"], dtype=torch.float32)
        y_eval_tensor = torch.tensor(data["y_eval"], dtype=torch.float32).view(-1, 1)
        x_test_tensor = torch.tensor(data["X_test"], dtype=torch.float32)
        y_test_tensor = torch.tensor(data["y_test"], dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        eval_dataset = TensorDataset(x_eval_tensor, y_eval_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False)

        return train_loader, eval_loader, test_loader
    