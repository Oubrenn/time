import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-6)


def _reduce_multivariate(x: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    if x.dim() == 1:
        return x
    if x.dim() != 2:
        raise ValueError("Expected 1D or 2D tensor for time series input.")
    if reduce == "mean":
        return x.mean(dim=0)
    if reduce == "first":
        return x[0]
    if reduce == "sum":
        return x.sum(dim=0)
    raise ValueError(f"Unknown reduce mode: {reduce}")


def build_views(x: torch.Tensor, view_config: ViewConfig) -> Dict[str, torch.Tensor]:
    x = _normalize(x)
    x_freq = stft_magnitude(x, n_fft=view_config.n_fft, hop_length=view_config.hop_length)
    x_tf = x_freq.log1p()

    shift_views = []
    shift_meta = []
    for bins in view_config.shift_bins:
        shifted = band_shift_stft(x_freq, bins)
        shift_views.append(shifted)
        shift_meta.append(ShiftParams(bins=bins))

    scale_views = []
    scale_meta = []
    for ratio in view_config.scale_ratios:
        scaled = frequency_scale_time(x, ratio)
        scale_views.append(scaled)
        scale_meta.append(ScaleParams(ratio=ratio))

    color_gains = make_coloring_gains(
        num_bins=x_freq.shape[0],
        bands=view_config.color_bands,
        max_gain_db=view_config.color_max_gain_db,
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return build_views(self.base[idx], self.view_config)


def _read_uea_ts(path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    class_labels: Optional[List[str]] = None
    data: List[List[List[float]]] = []
    labels: List[str] = []
    in_data = False

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if not in_data:
                if lower.startswith("@classlabel"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[1].lower() == "true":
                        class_labels = parts[2:]
                if lower.startswith("@data"):
                    in_data = True
                continue

            if class_labels is not None:
                *series_parts, label = line.split(":")
                labels.append(label.strip())
            else:
                series_parts = line.split(":")

            dims: List[List[float]] = []
            for part in series_parts:
                values = [float(v) for v in part.split(",") if v != ""]
                dims.append(values)
            data.append(dims)

    if not data:
        raise ValueError(f"No data found in {path}")

    num_dims = len(data[0])
    max_len = max(len(dim) for sample in data for dim in sample)
    tensor = torch.zeros(len(data), num_dims, max_len)
    for i, sample in enumerate(data):
        if len(sample) != num_dims:
            raise ValueError("Inconsistent number of dimensions across samples.")
        for d, values in enumerate(sample):
            length = len(values)
            if length:
                tensor[i, d, :length] = torch.tensor(values)

    if class_labels is None:
        return tensor, None

    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    label_tensor = torch.tensor([label_map[label] for label in labels], dtype=torch.long)
    return tensor, label_tensor


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def load_csv_series(path: Path) -> torch.Tensor:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            values = [_parse_float(item) for item in row]
            numeric = [v for v in values if v is not None]
            if numeric:
                rows.append(numeric)

    if not rows:
        raise ValueError(f"No numeric data found in {path}")

    if len(rows) == 1:
        return torch.tensor(rows[0], dtype=torch.float)

    if all(len(row) == len(rows[0]) for row in rows):
        matrix = torch.tensor(rows, dtype=torch.float).T
        return matrix.mean(dim=0)

    flattened = [value for row in rows for value in row]
    return torch.tensor(flattened, dtype=torch.float)


def _load_m4_series_list(path: Path) -> List[torch.Tensor]:
    series_list: List[torch.Tensor] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            values = [_parse_float(item) for item in row]
            numeric = [v for v in values if v is not None]
            if numeric:
                series_list.append(torch.tensor(numeric, dtype=torch.float))
    if not series_list:
        raise ValueError(f"No numeric series found in {path}")
    return series_list


class SlidingWindowDataset(Dataset):
    """Create sliding windows over a single time series."""

    def __init__(
        self,
        series: torch.Tensor,
        window: int,
        stride: int = 1,
        view_config: Optional[ViewConfig] = None,
    ) -> None:
        if series.dim() != 1:
            raise ValueError("SlidingWindowDataset expects a 1D tensor.")
        if window <= 0:
            raise ValueError("window must be positive.")
        self.series = series
        self.window = window
        self.stride = stride
        self.view_config = view_config or ViewConfig()
        self.starts = list(range(0, len(series) - window + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.starts[idx]
        x = self.series[start : start + self.window]
        return build_views(x, self.view_config)


class M4WindowDataset(Dataset):
    """Sliding windows over multiple M4-style series stored in CSV rows."""

    def __init__(
        self,
        path: str,
        window: int,
        stride: int = 1,
        view_config: Optional[ViewConfig] = None,
    ) -> None:
        self.series_list = _load_m4_series_list(Path(path))
        self.window = window
        self.stride = stride
        self.view_config = view_config or ViewConfig()

        self.index: List[Tuple[int, int]] = []
        for series_idx, series in enumerate(self.series_list):
            for start in range(0, len(series) - window + 1, stride):
                self.index.append((series_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        series_idx, start = self.index[idx]
        series = self.series_list[series_idx]
        x = series[start : start + self.window]
        return build_views(x, self.view_config)


class UEAClassificationDataset(Dataset):
    """Dataset loader for UEA/UCR .ts classification datasets."""

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "train",
        view_config: Optional[ViewConfig] = None,
        reduce: str = "mean",
    ) -> None:
        self.root = Path(root)
        self.name = name
        self.split = split.lower()
        self.view_config = view_config or ViewConfig()
        self.reduce = reduce

        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        file_name = f"{name}_{self.split.upper()}.ts"
        path = self.root / name / file_name
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset file at {path}")
        self.data, self.labels = _read_uea_ts(path)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = _reduce_multivariate(self.data[idx], reduce=self.reduce)
        views = build_views(x, self.view_config)
        if self.labels is not None:
            views["y"] = self.labels[idx]
        return views
