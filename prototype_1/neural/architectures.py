import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_layer_size: int = 39,
        hidden_layers_size: list[int] = [39, 21, 12],
        output_layer_size: int = 1,
        dropout_prob: float = 0.3,
    ) -> None:
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        current_layer_size = input_layer_size
        for layer_size in hidden_layers_size:
            self.layers.append(nn.Linear(current_layer_size, layer_size))
            self.layers.append(nn.LayerNorm(layer_size))
            self.layers.append(nn.ReLU(inplace=True))
            # self.layers.append(nn.Dropout(p=dropout_prob))
            current_layer_size = layer_size
        self.layers.append(nn.Linear(current_layer_size, output_layer_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
