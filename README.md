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

## Using real datasets (UEA/UCR .ts)
This repo includes a lightweight loader for UEA/UCR `.ts` classification datasets such as:
- SpokenArabicDigits
- JapaneseVowels
- UWaveGestureLibrary
- Heartbeat
- SelfRegulationSCP1
- SelfRegulationSCP2

Expected layout (example for SpokenArabicDigits):
```
data/SpokenArabicDigits/SpokenArabicDigits_TRAIN.ts
data/SpokenArabicDigits/SpokenArabicDigits_TEST.ts
```

Run:
```bash
python src/train.py --dataset uea --uea-name SpokenArabicDigits --data-dir data
```

## Forecasting datasets (ETT-small, M4)
For ETT-small, point `--csv-path` at the raw CSV file (e.g., `ETTh1.csv`). The loader will ignore non-numeric columns (like timestamps), then build sliding windows.

```bash
python src/train.py --dataset csv --csv-path data/ETT-small/ETTh1.csv --window-length 512 --window-stride 64
```

For M4, provide a CSV where each row is a series (the first column can be an ID). The loader builds windows across all rows.
```bash
python src/train.py --dataset m4 --csv-path data/M4/Hourly-train.csv --window-length 512 --window-stride 64
```

## Tests
```bash
pytest -q
```
