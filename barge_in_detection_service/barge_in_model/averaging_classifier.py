import torch
from torch import nn
import torch.nn.functional as F

class AverageModel(nn.Module):
    def __init__(self, audio_embedding_dim=768, hidden_dims=[768, 512, 256], output_dim=1, dropout_rate=0):
        super(AverageModel, self).__init__()

        layers = []
        prev_dim = audio_embedding_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim

        self.model = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, audio_embedding):
        out = self.model(audio_embedding)
        return self.output_layer(out)