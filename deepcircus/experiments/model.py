import torch
import torch.nn as nn

# ── Model ─────────────────────────────────────────────────────────────────────

class MLPDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            # MLP layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # MLP layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Logistic regression head
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)  # raw logits