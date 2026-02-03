import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ScaleParams:
    ratio: float


@dataclass(frozen=True)
class ShiftParams:
    bins: int


@dataclass(frozen=True)
class ColorParams:
    gains: torch.Tensor  # (bands,) linear gains


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    pad = target_len - x.shape[-1]
    return F.pad(x, (0, pad))


def frequency_scale_time(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """Resample in time to simulate speed/sampling-rate changes, then restore length."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    length = x.shape[-1]
    scaled_len = max(1, int(round(length / ratio)))
    x_in = x.unsqueeze(0).unsqueeze(0)
    x_scaled = F.interpolate(x_in, size=scaled_len, mode="linear", align_corners=False)
    x_scaled = x_scaled.squeeze(0).squeeze(0)
    return _match_length(x_scaled, length)


def band_shift_stft(stft: torch.Tensor, shift_bins: int) -> torch.Tensor:
    """Shift frequency bins in an STFT magnitude tensor (freq, time)."""
    return torch.roll(stft, shifts=shift_bins, dims=0)


def spectral_coloring(x: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """Apply spectral coloring in the frequency domain using linear gains."""
    spectrum = torch.fft.rfft(x)
    gains = gains.to(spectrum.device)
    if gains.numel() != spectrum.numel():
        gains = F.interpolate(
            gains.view(1, 1, -1), size=spectrum.numel(), mode="linear", align_corners=False
        ).view(-1)
    colored = spectrum * gains
    return torch.fft.irfft(colored, n=x.shape[-1])


def make_coloring_gains(num_bins: int, bands: int, max_gain_db: float) -> torch.Tensor:
    """Create a smooth random EQ curve in linear gains."""
    if bands <= 0:
        raise ValueError("bands must be positive")
    random_db = torch.empty(bands).uniform_(-max_gain_db, max_gain_db)
    random_linear = torch.pow(10.0, random_db / 20.0)
    gains = F.interpolate(
        random_linear.view(1, 1, -1),
        size=num_bins,
        mode="linear",
        align_corners=True,
    ).view(-1)
    return gains


def stft_magnitude(x: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    return stft.abs()


def spectral_centroid(magnitude: torch.Tensor) -> torch.Tensor:
    """Compute spectral centroid along frequency axis."""
    freq_bins = torch.arange(magnitude.shape[0], device=magnitude.device, dtype=magnitude.dtype)
    weights = magnitude.sum(dim=-1) + 1e-8
    centroid = (freq_bins * magnitude.sum(dim=-1)).sum() / weights.sum()
    return centroid


def build_meta(scale_params: ScaleParams, shift_params: ShiftParams, color_params: ColorParams) -> Dict[str, torch.Tensor]:
    return {
        "scale_ratio": torch.tensor(scale_params.ratio),
        "shift_bins": torch.tensor(shift_params.bins),
        "color_gains": color_params.gains.clone(),
    }
