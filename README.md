# Temperature Forecasting 24 Hours Ahead (Jena Climate 2009-2016)

A portfolio project in Earth-system-style data science: predict air temperature
24 hours ahead from historical weather measurements, using classic machine learning
(no deep learning), with an honest, time-respecting evaluation and quantified
uncertainty.

## 1. Problem

Given hourly weather measurements up to and including right now, predict what the
air temperature will be 24 hours from now, with a prediction interval, not just a
single number.

## 2. Data source and license note

- **Dataset:** Jena Climate 2009-2016, recorded every 10 minutes at the weather
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
3. Split by time: train 2009-2014, validation 2015, test 2016 (not a random split
   - see notebook for why that would leak information).
4. Trained a persistence baseline, Ridge regression, Random Forest, and
   HistGradientBoosting, compared with MAE/RMSE.
5. Quantified uncertainty with quantile gradient boosting (10th/50th/90th
   percentile), checked against actual test-set coverage.
6. Analyzed permutation importance and when/where the model fails (by month and
   by temperature extremity).

## 4. Results

All metrics are in degrees Celsius.

| Model                | Val MAE | Val RMSE | Test MAE | Test RMSE |
|-----------------------|--------:|---------:|---------:|----------:|
| Persistence baseline  |   2.757 |    3.584 |    2.469 |     3.238 |
| Ridge regression      |   2.484 |    3.160 |    2.259 |     2.880 |
| Random Forest         |   2.493 |    3.160 |    2.317 |     2.993 |
| HistGradientBoosting  |   2.415 |    3.056 |    2.301 |     2.924 |

Every trained model beats the persistence baseline on both MAE and RMSE. Ridge
regression and HistGradientBoosting come out nearly tied - the dominant signal
(current temperature + time-of-year cycle) is close to linear, so the extra
flexibility of tree ensembles buys only a small improvement here.

**Uncertainty (quantile gradient boosting, test set):**
- Target 80% interval coverage: 80.0%
- Measured coverage: 73.7% - the interval is a little narrow (model is slightly
  overconfident); average interval width was 6.44 degC.

**Feature importance:** current temperature dominates by far (~0.85), followed by
the yearly cosine cycle (~0.11) - knowing the season matters a lot for a
24-hour-ahead forecast. Pressure and the daily sine/cosine cycle come next, a
distant third and fourth.

**When does the model fail?** Hardest months were January, December, and August
(sharpest cold snaps / heat waves); easiest were March and October (gradual
seasonal transitions). Error on the coldest/hottest 1% of test days ran roughly
2-2.5x the overall MAE - the model is noticeably worse at genuine extremes than
at typical days.

## 5. Limitations

- Tested on a single out-of-sample year (2016) only.
- The 80% prediction interval under-covers slightly, so it's a bit overconfident.
- Clearly worse on extreme hot/cold days than on typical days.
- Single weather station - won't generalize to other locations without retraining.
- No deep learning tried, by design.

## 6. How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/temperature_forecasting.ipynb
```

## 7. Folder structure
|-- README.md
|-- requirements.txt
|-- data/jena_climate_2009_2016.csv
|-- figures/
`-- notebooks/temperature_forecasting.ipynb
