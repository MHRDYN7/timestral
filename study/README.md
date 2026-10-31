# Real-world Timestral forecasts

Six zero-shot forecasts from Timestral on public hourly weather. Everything added for this study lives in `study/`. The library, the tests, and the repository `README.md` are unchanged.

`study/new.ipynb` is the exploratory notebook from the local IDE (`Desktop\work\timestral\new.ipynb`), copied as it was. That notebook uses synthetic series. The six figures below are the real-data study, drawn the same way: the last 400 context hours, the median, and a band from quantile 0.1 to 0.9.

## Model

The script loads `TimestralForecaster.from_pretrained("mistralai/Timestral-209M")`. This machine has no CUDA device, so the weights ran on CPU. Each of the six calls finished in about 0.2–1.0 seconds after the load, so the study stayed on Timestral-209M and did not fall back to Timestral-28M.

`forecast.quantiles` has shape `(series, variates, horizon, 21)`. The 21 `quantile_levels` are `(0.01, 0.05, 0.1, …, 0.9, 0.95, 0.99)`. `forecast.median` is the 0.5 slice. `forecast.quantile(level)` accepts only one of those 21 levels. Every shaded band is quantile 0.1 to quantile 0.9.

## Series, variates, and covariates

A context array is `(series, variates, time)`. A single variate can be `(series, time)`. One series can be `(time,)`.

**Series** is the batch axis. The series in a batch are forecast together and do not attend to each other. Example 2 stacks Shanghai and London as shape `(2, 1, 2000)` in one call. The Shanghai median from that call matches the univariate Shanghai forecast.

**Variates** are channels of one series. They are forecast jointly. Example 3 is Shanghai temperature, humidity, and wind speed, shape `(1, 3, 2000)`.

**Covariates** are extra rows. They are not forecast, and covariate rows never read the targets. NaN is a missing value.

- `past_covariates` has shape `(series, covariates, time)` and is known on the history only.
- `known_covariates` has shape `(series, covariates, time + horizon)` and is known on the history and the future. A known-future row whose observed values are all 0 or 1 is an indicator: its location stays 0 and its scale stays 1.
- Categorical covariates are integer codes. NaN means missing. On a univariate target, each code is target-encoded per series: category `c` becomes `(series_mean + sum_c) / (1 + count_c)`. A code that never appears in the context maps to the series mean. A multivariate target receives the raw codes. Encoded rows are appended after the numeric covariates.

A pure random walk has a flat median at the last value and a band that widens like the square root of the lead time, because the mean increment is 0. A repeating pattern in the history moves the median. The notebook shows both on synthetic data: two cumulative random walks in one cell, and one series of three variates built from a 47-step double bump in another, where the median follows the bumps. Real winter temperature also repeats, once per day. The univariate Shanghai median moves by about 7.8°C across the 96-hour horizon instead of sitting on the last observation (2.5°C at 2023-12-31 23:00 UTC). By the last hour the 0.1–0.9 band runs from about −5.9°C to 8.7°C.

## Data

The numbers come from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). No API key. `study/run_examples.py` downloads them. The request URLs, the grid point the API actually returned, and the units are in `study/data/source.json`. The hours used for the forecasts are in `study/data/open_meteo_hourly.csv`.

| Place | Requested point | Grid point returned | Elevation |
| --- | --- | --- | --- |
| Shanghai | 31.2304°N, 121.4737°E | 31.247803°N, 121.5°E | 5 m |
| London | 51.5072°N, 0.1276°W | 51.493847°N, 0.1630249°W | 16 m |

| Column | Meaning |
| --- | --- |
| `time_utc` | Hour start in UTC, from the API field `hourly.time` |
| `*_temperature_2m_c` | Air temperature at 2 m, degrees Celsius |
| `*_relative_humidity_2m_pct` | Relative humidity at 2 m, percent |
| `*_wind_speed_10m_kmh` | Wind speed at 10 m, kilometres per hour |

Hour of day, the weekend flag, and day of week are not API columns. The script derives them from `time_utc`.

Context is 2000 hours, from 2023-10-09 16:00 UTC through 2023-12-31 23:00 UTC. That is the context length used in the notebook: long enough for many daily cycles and the cooling into winter, and short enough that these CPU forecasts finish immediately.

The horizon is 96 hours, 2024-01-01 00:00 UTC through 2024-01-04 23:00 UTC. Hourly temperature’s strong cycle is 24 hours, so 96 steps is four days and is the horizon in the notebook’s multivariate plot. A series whose main cycle is a week, such as hourly road traffic, would want a longer horizon. Here 96 hours is enough to see the daily rise and fall repeat. Each figure shows the last 400 context hours, the median, the 0.1–0.9 band, and a dashed line for what was recorded after the forecast origin.

## What each figure shows

### 1. Univariate series

`study/figures/01_univariate.png`

Shanghai air temperature alone, passed as a length-2000 vector. The forecaster treats that as one series and one variate. `quantiles` is `(1, 1, 96, 21)`. The median keeps the daily peak and trough, and the band widens with lead time.

### 2. Two independent series

`study/figures/02_two_independent_series.png`

Shanghai and London temperatures in one call, array shape `(2, 1, 2000)`. The two panels are separate series. London is the milder one: the last context hour is 7.7°C, and its median only ranges by about 3.1°C. The Shanghai panel matches example 1, including the same median path, because a series does not read the other series in the batch.

### 3. Joint multivariate series

`study/figures/03_multivariate.png`

One Shanghai series with three variates, shape `(1, 3, 2000)`: temperature, relative humidity, and wind speed. One subplot per variate. They are forecast together, so the temperature median is not the univariate median. The first forecast hour is 3.16°C here and 2.84°C in example 1.

### 4. Past-only covariate

`study/figures/04_past_covariate.png`

The target is Shanghai temperature again. Relative humidity is `past_covariates` with shape `(1, 1, 2000)`: known on the history, and not passed for the horizon. The lower panel stops the humidity line at the origin and shades the hours that were withheld. The first-hour temperature median moves to 3.01°C.

### 5. Known-future calendar covariate

`study/figures/05_known_future_covariate.png`

The same temperature target. `known_covariates` has shape `(1, 2, 2096)` and covers the 2000 history hours plus the 96 future hours.

- Row 0 is the UTC hour of day, 0–23. It is a numeric covariate.
- Row 1 is 1 on Saturday and Sunday UTC and 0 otherwise. Every value is 0 or 1, so this row is an indicator.

The lower panel draws both through 4 January. Hour of day keeps its daily sawtooth on the horizon. This horizon is Monday 1 January through Thursday 4 January 2024, so the weekend flag is 0 on every future hour. The first-hour temperature median is 2.94°C.

### 6. Categorical known covariate

`study/figures/06_categorical_covariate.png`

The same univariate temperature target, so target encoding runs. `categorical_known_covariates` is the day-of-week code, Monday=0 through Sunday=6, shape `(1, 1, 2096)`, including the horizon. Codes 0 through 6 all occur in the 2000-hour context. The horizon only uses Monday through Thursday (0–3), so each future code maps to that weekday’s smoothed mean temperature. The lower panel is the integer code and continues past the origin. The model receives the encoded values, appended after any numeric covariates. There are no numeric covariates in this call. The first-hour temperature median is 2.80°C.

## Notebook

`study/new.ipynb` is the local exploration, not one of the six weather examples. It forecasts two independent random-walk series, then one three-variate series built from a repeating 47-step double bump plus a small random walk. In that second plot the median follows the bumps, the shaded band is quantile 0.1 to 0.9, and the dashed line is the bump’s continuation with the walk held at its last value. One cell is leftover text (a list of attribute names) rather than a program. `study/run_examples.py` does not execute the notebook.

## Run it again

From the repository root, with Python 3.10 or newer:

```bash
pip install -e .
pip install matplotlib
python study/run_examples.py
```

The script downloads the weather, loads Timestral-209M on CUDA when `torch.cuda.is_available()` is true and on CPU otherwise, and rewrites the six PNGs in `study/figures/`. `study/data/forecast_summary.json` stores the quantile shapes and a few median values from the last run.
