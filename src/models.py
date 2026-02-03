from typing import Tuple

import torch
import torch.nn as nn


class ConvMambaEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mamba = self._build_mamba(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def _build_mamba(self, hidden_dim: int) -> nn.Module:
        try:
            from mamba_ssm import Mamba

            return Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        except Exception:
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.mamba(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


class MultiViewModel(nn.Module):
    def __init__(self, hidden_dim: int = 64, output_dim: int = 128) -> None:
        super().__init__()
        self.time_encoder = ConvMambaEncoder(1, hidden_dim, output_dim)
        self.freq_encoder = ConvMambaEncoder(1, hidden_dim, output_dim)

    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_time = self.time_encoder(x_time)
        z_freq = self.freq_encoder(x_freq)
        return z_time, z_freq
