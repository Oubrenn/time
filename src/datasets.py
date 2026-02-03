from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .transforms import (
    ColorParams,
    ScaleParams,
    ShiftParams,
    band_shift_stft,
    build_meta,
    frequency_scale_time,
    make_coloring_gains,
    spectral_coloring,
    stft_magnitude,
)


@dataclass(frozen=True)
class ViewConfig:
    n_fft: int = 256
    hop_length: int = 64
    shift_bins: List[int] = None
    scale_ratios: List[float] = None
    color_bands: int = 8
    color_max_gain_db: float = 6.0

    def __post_init__(self):
        object.__setattr__(self, "shift_bins", self.shift_bins or [3, -3])
        object.__setattr__(self, "scale_ratios", self.scale_ratios or [0.9, 1.1])


class SyntheticTimeSeriesDataset(Dataset):
    """Simple dataset for MVP validation with controllable transforms."""

    def __init__(
        self,
        num_samples: int,
        length: int,
        view_config: Optional[ViewConfig] = None,
    ) -> None:
        self.num_samples = num_samples
        self.length = length
        self.view_config = view_config or ViewConfig()
        self.base = torch.randn(num_samples, length)

    def __len__(self) -> int:
        return self.num_samples

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean()) / (x.std() + 1e-6)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self._normalize(self.base[idx])
        x_freq = stft_magnitude(x, n_fft=self.view_config.n_fft, hop_length=self.view_config.hop_length)
        x_tf = x_freq.log1p()

        shift_views = []
        shift_meta = []
        for bins in self.view_config.shift_bins:
            shifted = band_shift_stft(x_freq, bins)
            shift_views.append(shifted)
            shift_meta.append(ShiftParams(bins=bins))

        scale_views = []
        scale_meta = []
        for ratio in self.view_config.scale_ratios:
            scaled = frequency_scale_time(x, ratio)
            scale_views.append(scaled)
            scale_meta.append(ScaleParams(ratio=ratio))

        color_gains = make_coloring_gains(
            num_bins=x_freq.shape[0],
            bands=self.view_config.color_bands,
            max_gain_db=self.view_config.color_max_gain_db,
        )
        color_view = spectral_coloring(x, color_gains)
        color_meta = ColorParams(gains=color_gains)

        meta = {
            "shift": [build_meta(ScaleParams(1.0), sp, color_meta) for sp in shift_meta],
            "scale": [build_meta(sp, ShiftParams(0), color_meta) for sp in scale_meta],
            "color": build_meta(ScaleParams(1.0), ShiftParams(0), color_meta),
        }

        return {
            "x_time": x,
            "x_freq": x_freq,
            "x_tf": x_tf,
            "x_shift": shift_views,
            "x_scale": scale_views,
            "x_color": color_view,
            "meta": meta,
        }
