import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=5, hidden_sizes=(256, 64), output_size=5, activation='relu', init_type='xavier'):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        self._initialize_weights(init_type)
    
    def forward(self, x):
        return self.network(x)
    
    def _initialize_weights(self, init_type):
        for m in self.network:
            if isinstance(m, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                elif init_type == 'he':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)
                elif init_type == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.zeros_(m.bias)
                elif init_type == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                    nn.init.zeros_(m.bias)