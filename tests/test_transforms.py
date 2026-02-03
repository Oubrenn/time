import torch

from src.transforms import (
    frequency_scale_time,
    make_coloring_gains,
    spectral_centroid,
    stft_magnitude,
)


def test_frequency_scale_reproducible():
    x = torch.linspace(0, 1, 256)
    out1 = frequency_scale_time(x, ratio=1.2)
    out2 = frequency_scale_time(x, ratio=1.2)
    assert torch.allclose(out1, out2)


def test_coloring_gain_shape():
    gains = make_coloring_gains(num_bins=128, bands=4, max_gain_db=3.0)
    assert gains.shape == (128,)


def test_spectral_centroid_monotonic():
    x = torch.sin(torch.linspace(0, 10 * torch.pi, 512))
    mag = stft_magnitude(x, n_fft=128, hop_length=32)
    centroid_base = spectral_centroid(mag)
    x_scaled = frequency_scale_time(x, ratio=0.8)
    mag_scaled = stft_magnitude(x_scaled, n_fft=128, hop_length=32)
    centroid_scaled = spectral_centroid(mag_scaled)
    assert centroid_scaled > centroid_base
