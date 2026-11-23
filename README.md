# Temperature Forecasting 24 Hours Ahead (Jena Climate 2009–2016)

A portfolio project in Earth-system-style data science: predict air temperature
24 hours ahead from historical weather measurements, using classic machine learning
(no deep learning), with an honest, time-respecting evaluation and quantified
uncertainty.

## 1. Problem

Given hourly weather measurements up to and including right now, predict what the
air temperature will be 24 hours from now, with a prediction interval, not just a
single number.

## 2. Data source and license note

- **Dataset:** Jena Climate 2009–2016, recorded every 10 minutes at the weather
  station on the roof of the Max Planck Institute for Biogeochemistry, Jena, Germany.
- **License:** no explicit license is attached to this pre-packaged CSV. It is
  widely reused for educational/research purposes (TensorFlow/Keras tutorials).
  Check the institute's own weather data page before any commercial use.

## 3. Method

1. Loaded and cleaned the data: parsed timestamps, resampled to hourly, replaced
   -9999 sensor-error codes with NaN and interpolated them, dropped 24 duplicate
   rows from a known recording glitch.
2. Engineered features: wind x/y components, sine/cosine time-of-day and
   time-of-year encodings, lag features (1/3/6/12/24h), 6h/24h rolling means.
3. Split by time: train 2009–2014, validation 2015, test 2016 (not a random split
   — see notebook for why that would leak information).
4. Trained a persistence baseline, Ridge regression, Random Forest, and
   HistGradientBoosting, compared with MAE/RMSE.
5. Quantified uncertainty with quantile gradient boosting (10th/50th/90th
   percentile), checked against actual test-set coverage.
6. Analyzed permutation importance and when/where the model fails (by month and
   by temperature extremity).

## 4. Results

(fill in your results_table.csv numbers here as a markdown table)

## 5. Limitations

- Tested on a single out-of-sample year (2016) only.
- The 80% prediction interval under-covers slightly, so it's a bit overconfident.
- Clearly worse on extreme hot/cold days than on typical days.
- Single weather station — won't generalize to other locations without retraining.
- No deep learning tried, by design.

## 6. How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/temperature_forecasting.ipynb
```

## 7. Folder structure
├── README.md
├── requirements.txt
├── data/jena_climate_2009_2016.csv
├── figures/
└── notebooks/temperature_forecasting.ipynb
