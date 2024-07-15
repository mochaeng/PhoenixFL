import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_layer_size: int = 39,
        # hidden_layers_size: list[int] = [512, 384, 256, 128, 64],
        hidden_layers_size: list[int] = [256, 128, 64],
        output_layer_size: int = 1,
        dropout_prob: float = 0.2,
    ) -> None:
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        current_layer_size = input_layer_size
        for layer_size in hidden_layers_size:
            self.layers.append(nn.Linear(current_layer_size, layer_size))
            self.layers.append(nn.LayerNorm(layer_size))
            # self.layers.append(nn.BatchNorm1d(layer_size))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(p=dropout_prob))
            current_layer_size = layer_size
        self.layers.append(nn.Linear(current_layer_size, output_layer_size))

        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
