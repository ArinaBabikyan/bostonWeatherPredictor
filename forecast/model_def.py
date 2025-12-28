import torch
import torch.nn as nn

class WeatherGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(16, 1)


    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)
