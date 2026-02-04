from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    """
    Pure PyTorch implementation of Mamba for compatibility.
    This is a simplified version that avoids CUDA compilation issues.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # Selective parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + d_conv * 2, bias=False)

        # State space parameters
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_state, self.d_inner)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) where B is batch size, L is sequence length, D is d_model

        Returns:
            output: (B, L, D)
        """
        B, L, D = x.shape

        # Input projection
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # 1D Convolution for local patterns
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :L]  # (B, d_inner, L)
        x = x.transpose(1, 2)  # (B, L, d_inner)

        # Selective SSM
        y = self._selective_scan(x)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)
        return output

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        """
        Simplified selective state space model scan.
        Args:
            u: (B, L, d_inner)
        Returns:
            y: (B, L, d_inner)
        """
        B, L, d_inner = u.shape
        d_state = self.d_state

        # Compute selective parameters
        A = -torch.exp(self.A_log.float())  # (d_state, d_inner)
        D = self.D

        # Initialize state
        state = torch.zeros(B, d_state, d_inner, dtype=u.dtype, device=u.device)

        outputs = []

        # Scan over sequence
        for t in range(L):
            u_t = u[:, t, :]  # (B, d_inner)

            # Selective scan (simplified)
            # y_t = A @ state + u_t
            # state = A @ state + u_t
            state = state * A.unsqueeze(0) + u_t.unsqueeze(1)  # (B, d_state, d_inner)

            # Output
            y_t = torch.sum(state, dim=1) + D * u_t  # (B, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return y


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
        # Use built-in pure PyTorch Mamba implementation
        return Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)

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
