import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from datasets import SyntheticTimeSeriesDataset, ViewConfig
from losses import ta_cfc_loss, tf_consistency_loss
from models import MultiViewModel
from transforms import stft_magnitude


def collate_fn(batch):
    x_time = torch.stack([b["x_time"] for b in batch])
    x_freq = torch.stack([b["x_freq"] for b in batch])
    x_scale = torch.stack([b["x_scale"][0] for b in batch])
    x_color = torch.stack([b["x_color"] for b in batch])
    return {
        "x_time": x_time,
        "x_freq": x_freq,
        "x_scale": x_scale,
        "x_color": x_color,
    }


def run_epoch(model, loader, optimizer=None, device="cpu", ta_mode="vicreg"):
    total = 0.0
    for batch in loader:
        x_time = batch["x_time"].to(device)
        x_freq = batch["x_freq"].to(device)
        x_scale = batch["x_scale"].to(device)
        x_color = batch["x_color"].to(device)

        z_time, z_freq = model(x_time, x_freq)
        x_scale_freq = stft_magnitude(x_scale, n_fft=256, hop_length=64)
        z_scale, _ = model(x_scale, x_scale_freq)
        x_color_freq = stft_magnitude(x_color, n_fft=256, hop_length=64)
        z_color, _ = model(x_color, x_color_freq)

        loss_tf = tf_consistency_loss(z_time, z_freq)
        loss_ta = ta_cfc_loss(z_time, z_scale, mode=ta_mode) + ta_cfc_loss(z_time, z_color, mode=ta_mode)
        loss = loss_tf + loss_ta

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total += loss.item()
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ta-mode", type=str, default="vicreg", choices=["vicreg", "infonce"])
    args = parser.parse_args()

    view_config = ViewConfig()
    dataset = SyntheticTimeSeriesDataset(num_samples=200, length=1024, view_config=view_config)
    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiViewModel().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device=args.device, ta_mode=args.ta_mode)
        val_loss = run_epoch(model, val_loader, device=args.device, ta_mode=args.ta_mode)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    test_loss = run_epoch(model, test_loader, device=args.device, ta_mode=args.ta_mode)
    print(f"test_loss={test_loss:.4f}")


if __name__ == "__main__":
    main()
