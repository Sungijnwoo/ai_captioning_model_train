from torch import nn
import torch


class ClipToLlmProjector(nn.Module):
    def __init__(self, clip_dim: int, llm_dim: int, multi_rate: int) -> None:
        super().__init__()
        hidden = llm_dim * multi_rate
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)