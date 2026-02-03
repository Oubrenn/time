# time

Minimal MVP for spectral-shift robustness experiments in time series.

## What is included
- Dataset that returns multi-view dict (time, frequency, time-frequency, shift/scale/color variants) for rapid experimentation.
- Transform operators for frequency scaling, band translation, and spectral coloring.
- Losses for TF-Consistency and TA-CFC (VICReg/InfoNCE).
- Minimal training/validation/test loop with logging.

## Quick start
```bash
python src/train.py --epochs 3 --batch-size 8
```

## Tests
```bash
pytest -q
```
