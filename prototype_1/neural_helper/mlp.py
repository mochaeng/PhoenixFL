import torch
from torch.utils.data import DataLoader, TensorDataset, dataset
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

from typing import Dict


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.BCEWithLogitsLoss
OPTIMIZER = torch.optim.SGD


class PopoolaMLP(nn.Module):
    def __init__(self) -> None:
        super(PopoolaMLP, self).__init__()
        self.fc1 = nn.Linear(41, 128)
        self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.norm2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.sigmoid(x)
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


def load_data(data: dict, batch_size=32):
    x_train_tensor = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train_tensor = torch.tensor(data["y_train"], dtype=torch.float32).view(-1, 1)
    x_eval_tensor = torch.tensor(data["X_eval"], dtype=torch.float32)
    y_eval_tensor = torch.tensor(data["y_eval"], dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test_tensor = torch.tensor(data["y_test"], dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    eval_dataset = TensorDataset(x_eval_tensor, y_eval_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_loader, eval_loader, test_loader


def train(net: nn.Module, trainloader, epochs: int=10, lr=1e-4, momentum=9e-1, weight_decay=1e-5, is_verbose=True):
    criterion = CRITERION()
    optimizer = OPTIMIZER(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss
            total += labels.size(0)
            round_outputs = torch.round(torch.sigmoid(outputs))
            correct += (round_outputs == labels).sum().item()
            
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        
        if is_verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
            

def test_metrics(net: nn.Module, testloader) -> Dict[str, float]:
    criterion = CRITERION()
    loss = 0
    metrics: Dict[str, float] = {}
    
    accuracy = BinaryAccuracy().to(DEVICE)
    precision = BinaryPrecision().to(DEVICE)
    recall = BinaryRecall().to(DEVICE)
    f1_score = BinaryF1Score().to(DEVICE)
    
    net.eval()
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            predicted = torch.round(torch.sigmoid(outputs.data))
            
            labels = labels.bool()
            predicted = predicted.view(-1)
            labels = labels.view(-1)
            
            accuracy.update(predicted, labels)
            precision.update(predicted, labels)
            recall.update(predicted, labels)
            f1_score.update(predicted, labels)
    
    loss /= len(testloader.dataset)
    
    metrics = {
        "accuracy": accuracy.compute().item(),
        "precision": precision.compute().item(),
        "recall": recall.compute().item(),
        "f1_score": f1_score.compute().item(),
        "final_loss": loss,
    }
    
    return metrics
    