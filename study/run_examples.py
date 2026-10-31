"""Regenerate the six real-world Timestral figures in study/figures/.

Downloads hourly weather from the Open-Meteo archive (no API key), forecasts with
Timestral-209M on CUDA when a GPU is present and on CPU otherwise, and writes
one PNG per example. The plotting follows the local notebook study/new.ipynb:
the last 400 context steps, the median, and the quantile 0.1–0.9 band.

Run from the repository root, with this package installed:

    python study/run_examples.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from timestral import TimestralForecaster

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

MODEL_ID = "mistralai/Timestral-209M"
CONTEXT = 2000  # same context length as study/new.ipynb
HORIZON = 96  # four days; cell 7 of the notebook uses this horizon
TAIL = 400
ORIGIN = "2024-01-01T00:00"  # first hour of the horizon, UTC
DOWNLOAD_START = "2023-10-01"
DOWNLOAD_END = "2024-01-06"

# Open-Meteo snaps the request to its weather grid and returns that point.
PLACES = {
    "shanghai": {"latitude": 31.2304, "longitude": 121.4737, "label": "Shanghai"},
    "london": {"latitude": 51.5072, "longitude": -0.1276, "label": "London"},
}
HOURLY_VARS = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m")
ARCHIVE_HOSTS = (
    "https://archive-api.open-meteo.com/v1/archive",
    "https://api.open-meteo.com/v1/archive",
)

VAR_LABEL = {
    "temperature_2m": "air temperature at 2 m (°C)",
    "relative_humidity_2m": "relative humidity at 2 m (%)",
    "wind_speed_10m": "wind speed at 10 m (km/h)",
}


def fetch_place(place_id: str) -> tuple[dict, str]:
    """Download one place. Tries the archive host, then the public API host."""
    spec = PLACES[place_id]
    query = (
        f"?latitude={spec['latitude']}&longitude={spec['longitude']}"
        f"&start_date={DOWNLOAD_START}&end_date={DOWNLOAD_END}"
        f"&hourly={','.join(HOURLY_VARS)}&timezone=UTC"
    )
    errors: list[str] = []
    for host in ARCHIVE_HOSTS:
        url = host + query
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "timestral-study/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            hourly = payload.get("hourly") or {}
            if not hourly.get("time") or any(name not in hourly for name in HOURLY_VARS):
                raise ValueError(f"{url} did not return the requested hourly columns")
            return payload, url
        except Exception as exc:  # noqa: BLE001 - try the next public host
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Open-Meteo download failed:\n" + "\n".join(errors))


def _column(values: list) -> np.ndarray:
    return np.array([np.nan if value is None else float(value) for value in values], dtype=np.float64)


def align_places(payloads: dict[str, dict]) -> dict:
    """Inner-join the two cities on the UTC hour stamp the API returned."""
    frames = {}
    for place_id, payload in payloads.items():
        hourly = payload["hourly"]
        frames[place_id] = {
            "time": list(hourly["time"]),
            **{name: _column(hourly[name]) for name in HOURLY_VARS},
        }
    shanghai_time = frames["shanghai"]["time"]
    london_index = {stamp: i for i, stamp in enumerate(frames["london"]["time"])}
    shared = [stamp for stamp in shanghai_time if stamp in london_index]
    if ORIGIN not in shared:
        raise RuntimeError(f"{ORIGIN} is not in both downloaded series")
    origin_at = shared.index(ORIGIN)
    if origin_at < CONTEXT or origin_at + HORIZON > len(shared):
        raise RuntimeError(
            f"need {CONTEXT} hours before {ORIGIN} and {HORIZON} after it; "
            f"got index {origin_at} of {len(shared)}"
        )
    window = shared[origin_at - CONTEXT : origin_at + HORIZON]
    shanghai_index = {stamp: i for i, stamp in enumerate(shanghai_time)}

    def take(place_id: str, name: str) -> np.ndarray:
        frame = frames[place_id]
        index = shanghai_index if place_id == "shanghai" else london_index
        return np.array([frame[name][index[stamp]] for stamp in window], dtype=np.float64)

    times = [datetime.fromisoformat(stamp) for stamp in window]
    missing = {
        f"{place_id}.{name}": int(np.isnan(take(place_id, name)).sum())
        for place_id in payloads
        for name in HOURLY_VARS
    }
    if any(missing.values()):
        raise RuntimeError(f"downloaded window has missing hours: {missing}")
    return {
        "times": times,
        "stamps": window,
        "shanghai": {name: take("shanghai", name) for name in HOURLY_VARS},
        "london": {name: take("london", name) for name in HOURLY_VARS},
        "missing": missing,
    }


def calendar(times: list[datetime]) -> dict[str, np.ndarray]:
    """Quantities known from the civil clock, including hours that have not happened yet."""
    hour = np.array([stamp.hour for stamp in times], dtype=np.float64)
    # Monday is 0 and Sunday is 6, matching datetime.weekday.
    day_of_week = np.array([stamp.weekday() for stamp in times], dtype=np.float64)
    weekend = (day_of_week >= 5).astype(np.float64)
    return {"hour_of_day": hour, "day_of_week": day_of_week, "weekend": weekend}


def save_table(bundle: dict, sources: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    header = [
        "time_utc",
        "shanghai_temperature_2m_c",
        "shanghai_relative_humidity_2m_pct",
        "shanghai_wind_speed_10m_kmh",
        "london_temperature_2m_c",
        "london_relative_humidity_2m_pct",
        "london_wind_speed_10m_kmh",
    ]
    lines = [",".join(header)]
    for i, stamp in enumerate(bundle["stamps"]):
        row = [
            stamp,
            f"{bundle['shanghai']['temperature_2m'][i]:.4f}",
            f"{bundle['shanghai']['relative_humidity_2m'][i]:.4f}",
            f"{bundle['shanghai']['wind_speed_10m'][i]:.4f}",
            f"{bundle['london']['temperature_2m'][i]:.4f}",
            f"{bundle['london']['relative_humidity_2m'][i]:.4f}",
            f"{bundle['london']['wind_speed_10m'][i]:.4f}",
        ]
        lines.append(",".join(row))
    (DATA_DIR / "open_meteo_hourly.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DATA_DIR / "source.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")


def arrays(forecast, series: int, variate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = forecast.median[series, variate].detach().cpu().numpy()
    low = forecast.quantile(0.1)[series, variate].detach().cpu().numpy()
    high = forecast.quantile(0.9)[series, variate].detach().cpu().numpy()
    return median, low, high


def draw_forecast(ax, times_hist, history, times_fut, median, low, high, actual, ylabel: str) -> None:
    """Same layers as cell 7 of study/new.ipynb, on a real time axis."""
    ax.plot(times_hist, history, label="recorded history")
    ax.plot(times_fut, median, label="median")
    ax.fill_between(times_fut, low, high, alpha=0.25, label="q10–q90")
    ax.plot(times_fut, actual, ls="--", color="0.3", label="recorded afterward")
    ax.axvline(times_fut[0], color="0.5", lw=0.8, ls=":")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def note_from(forecast, seconds: float, quantiles_shape: list[int]) -> dict:
    median, low, high = arrays(forecast, 0, 0)
    return {
        "seconds": round(seconds, 2),
        "quantiles_shape": quantiles_shape,
        "median_hour_0": round(float(median[0]), 3),
        "median_hour_24": round(float(median[24]), 3),
        "median_hour_48": round(float(median[48]), 3),
        "median_hour_95": round(float(median[-1]), 3),
        "q10_hour_95": round(float(low[-1]), 3),
        "q90_hour_95": round(float(high[-1]), 3),
        "median_range": round(float(median.max() - median.min()), 3),
    }


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={MODEL_ID}", flush=True)

    payloads = {}
    urls = {}
    for place_id in PLACES:
        payloads[place_id], urls[place_id] = fetch_place(place_id)
        returned = payloads[place_id]
        print(
            f"downloaded {place_id}: grid=({returned['latitude']}, {returned['longitude']}) "
            f"elevation_m={returned.get('elevation')} hours={len(returned['hourly']['time'])}",
            flush=True,
        )
    bundle = align_places(payloads)
    clock = calendar(bundle["times"])
    context_times = bundle["times"][:CONTEXT]
    future_times = bundle["times"][CONTEXT:]
    hist_times = context_times[-TAIL:]
    sources = {
        "api": "Open-Meteo Historical Weather API",
        "documentation": "https://open-meteo.com/en/docs/historical-weather-api",
        "requests": urls,
        "download_start": DOWNLOAD_START,
        "download_end": DOWNLOAD_END,
        "forecast_origin_utc": ORIGIN,
        "context_hours": CONTEXT,
        "horizon_hours": HORIZON,
        "places": {
            place_id: {
                "requested_latitude": PLACES[place_id]["latitude"],
                "requested_longitude": PLACES[place_id]["longitude"],
                "grid_latitude": payloads[place_id]["latitude"],
                "grid_longitude": payloads[place_id]["longitude"],
                "elevation_m": payloads[place_id].get("elevation"),
                "hourly_units": payloads[place_id].get("hourly_units"),
            }
            for place_id in PLACES
        },
        "columns": {
            "time_utc": "Hour start in UTC, ISO-8601, from the API field hourly.time.",
            "temperature_2m": "Air temperature at 2 metres, degrees Celsius.",
            "relative_humidity_2m": "Relative humidity at 2 metres, percent.",
            "wind_speed_10m": "Wind speed at 10 metres, kilometres per hour.",
            "hour_of_day": "UTC hour 0–23, derived from time_utc. Known for any future hour.",
            "weekend": "1 on Saturday and Sunday UTC, else 0. A 0/1 indicator known ahead.",
            "day_of_week": "Monday=0 through Sunday=6, integer codes derived from time_utc.",
        },
    }
    save_table(bundle, sources)

    started = time.perf_counter()
    forecaster = TimestralForecaster.from_pretrained(MODEL_ID, device=device)
    print(f"loaded {MODEL_ID} on {device} in {time.perf_counter() - started:.1f}s", flush=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": MODEL_ID,
        "device": device,
        "cuda": bool(torch.cuda.is_available()),
        "context_hours": CONTEXT,
        "prediction_length": HORIZON,
        "quantile_levels": list(forecaster.quantile_levels),
        "examples": {},
    }

    shanghai_temp = bundle["shanghai"]["temperature_2m"]
    temp_context = shanghai_temp[:CONTEXT]
    temp_future = shanghai_temp[CONTEXT:]

    # 1. Univariate. A length-T vector is one series with one variate.
    t0 = time.perf_counter()
    forecast = forecaster.forecast(temp_context, prediction_length=HORIZON)
    elapsed = time.perf_counter() - t0
    assert tuple(forecast.quantiles.shape) == (1, 1, HORIZON, 21)
    median, low, high = arrays(forecast, 0, 0)
    fig, ax = plt.subplots(figsize=(12, 4))
    draw_forecast(
        ax,
        hist_times,
        temp_context[-TAIL:],
        future_times,
        median,
        low,
        high,
        temp_future,
        "°C",
    )
    ax.set_title("Univariate series — Shanghai air temperature at 2 m")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("UTC time")
    path = FIG_DIR / "01_univariate.png"
    finish(fig, path)
    summary["examples"]["univariate"] = note_from(forecast, elapsed, list(forecast.quantiles.shape))
    summary["examples"]["univariate"]["figure"] = str(path.relative_to(ROOT.parent))
    print(f"1 univariate {tuple(forecast.quantiles.shape)} {elapsed:.1f}s", flush=True)

    # 2. Two independent series in one call. No cross-series attention.
    batch = np.stack([temp_context, bundle["london"]["temperature_2m"][:CONTEXT]], axis=0)[:, None, :]
    assert batch.shape == (2, 1, CONTEXT)
    t0 = time.perf_counter()
    forecast = forecaster.forecast(batch, prediction_length=HORIZON)
    elapsed = time.perf_counter() - t0
    assert tuple(forecast.quantiles.shape) == (2, 1, HORIZON, 21)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    actuals = (temp_future, bundle["london"]["temperature_2m"][CONTEXT:])
    names = ("Shanghai air temperature at 2 m", "London air temperature at 2 m")
    histories = (temp_context, bundle["london"]["temperature_2m"][:CONTEXT])
    for i, ax in enumerate(axes):
        median, low, high = arrays(forecast, i, 0)
        draw_forecast(ax, hist_times, histories[i][-TAIL:], future_times, median, low, high, actuals[i], "°C")
        ax.set_title(names[i])
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("UTC time")
    fig.suptitle("Two independent series — shape (2, 1, 2000), one forecast call")
    path = FIG_DIR / "02_two_independent_series.png"
    finish(fig, path)
    summary["examples"]["two_independent_series"] = note_from(forecast, elapsed, list(forecast.quantiles.shape))
    summary["examples"]["two_independent_series"]["figure"] = str(path.relative_to(ROOT.parent))
    london_median = arrays(forecast, 1, 0)[0]
    summary["examples"]["two_independent_series"]["london_median_hour_0"] = round(float(london_median[0]), 3)
    summary["examples"]["two_independent_series"]["london_median_range"] = round(
        float(london_median.max() - london_median.min()), 3
    )
    print(f"2 batch {tuple(forecast.quantiles.shape)} {elapsed:.1f}s", flush=True)

    # 3. One series, three variates, forecast jointly.
    variate_names = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m")
    multi = np.stack([bundle["shanghai"][name][:CONTEXT] for name in variate_names], axis=0)[None, :, :]
    assert multi.shape == (1, 3, CONTEXT)
    t0 = time.perf_counter()
    forecast = forecaster.forecast(multi, prediction_length=HORIZON)
    elapsed = time.perf_counter() - t0
    assert tuple(forecast.quantiles.shape) == (1, 3, HORIZON, 21)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ylabels = ("°C", "%", "km/h")
    for i, ax in enumerate(axes):
        median, low, high = arrays(forecast, 0, i)
        actual = bundle["shanghai"][variate_names[i]][CONTEXT:]
        draw_forecast(
            ax,
            hist_times,
            multi[0, i, -TAIL:],
            future_times,
            median,
            low,
            high,
            actual,
            ylabels[i],
        )
        ax.set_title(VAR_LABEL[variate_names[i]])
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("UTC time")
    fig.suptitle("Joint multivariate series — Shanghai, shape (1, 3, 2000)")
    path = FIG_DIR / "03_multivariate.png"
    finish(fig, path)
    summary["examples"]["multivariate"] = note_from(forecast, elapsed, list(forecast.quantiles.shape))
    summary["examples"]["multivariate"]["figure"] = str(path.relative_to(ROOT.parent))
    print(f"3 multivariate {tuple(forecast.quantiles.shape)} {elapsed:.1f}s", flush=True)

    # 4. Past-only covariate: humidity is known on the history and withheld from the horizon.
    humidity = bundle["shanghai"]["relative_humidity_2m"][:CONTEXT][None, None, :]
    assert humidity.shape == (1, 1, CONTEXT)
    t0 = time.perf_counter()
    forecast = forecaster.forecast(temp_context, prediction_length=HORIZON, past_covariates=humidity)
    elapsed = time.perf_counter() - t0
    assert tuple(forecast.quantiles.shape) == (1, 1, HORIZON, 21)
    median, low, high = arrays(forecast, 0, 0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    draw_forecast(axes[0], hist_times, temp_context[-TAIL:], future_times, median, low, high, temp_future, "°C")
    axes[0].set_title("Shanghai air temperature, conditioned on past humidity only")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(hist_times, humidity[0, 0, -TAIL:], color="#1f4e79", label="relative humidity (past only)")
    axes[1].axvline(future_times[0], color="0.5", lw=0.8, ls=":")
    axes[1].axvspan(future_times[0], future_times[-1], color="0.85", label="withheld from the horizon")
    axes[1].set_ylabel("%")
    axes[1].set_xlabel("UTC time")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Past-only covariate — humidity known on the history, not on the horizon")
    path = FIG_DIR / "04_past_covariate.png"
    finish(fig, path)
    summary["examples"]["past_covariate"] = note_from(forecast, elapsed, list(forecast.quantiles.shape))
    summary["examples"]["past_covariate"]["figure"] = str(path.relative_to(ROOT.parent))
    print(f"4 past covariate {tuple(forecast.quantiles.shape)} {elapsed:.1f}s", flush=True)

    # 5. Known-future calendar rows cover the context and the horizon.
    full = CONTEXT + HORIZON
    known = np.stack([clock["hour_of_day"], clock["weekend"]], axis=0)[None, :, :]
    assert known.shape == (1, 2, full)
    t0 = time.perf_counter()
    forecast = forecaster.forecast(temp_context, prediction_length=HORIZON, known_covariates=known)
    elapsed = time.perf_counter() - t0
    assert tuple(forecast.quantiles.shape) == (1, 1, HORIZON, 21)
    median, low, high = arrays(forecast, 0, 0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    draw_forecast(axes[0], hist_times, temp_context[-TAIL:], future_times, median, low, high, temp_future, "°C")
    axes[0].set_title("Shanghai air temperature, with hour of day and a weekend flag known ahead")
    axes[0].legend(loc="upper left", fontsize=8)
    shown = slice(CONTEXT - TAIL, full)
    axes[1].plot(bundle["times"][shown], clock["hour_of_day"][shown], label="hour of day, UTC (0–23)")
    axes[1].plot(bundle["times"][shown], clock["weekend"][shown] * 23, ls="--", label="weekend flag, drawn ×23 so it is visible")
    axes[1].axvline(future_times[0], color="0.5", lw=0.8, ls=":")
    axes[1].set_ylabel("hour")
    axes[1].set_xlabel("UTC time")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Known-future calendar covariate — values continue through the horizon")
    path = FIG_DIR / "05_known_future_covariate.png"
    finish(fig, path)
    summary["examples"]["known_future"] = note_from(forecast, elapsed, list(forecast.quantiles.shape))
    summary["examples"]["known_future"]["figure"] = str(path.relative_to(ROOT.parent))
    print(f"5 known future {tuple(forecast.quantiles.shape)} {elapsed:.1f}s", flush=True)

    # 6. Categorical known covariate on a univariate target, so target encoding runs.
    codes = clock["day_of_week"][None, None, :]
    assert codes.shape == (1, 1, full)
    t0 = time.perf_counter()
    forecast = forecaster.forecast(temp_context, prediction_length=HORIZON, categorical_known_covariates=codes)
    elapsed = time.perf_counter() - t0
    assert tuple(forecast.quantiles.shape) == (1, 1, HORIZON, 21)
    median, low, high = arrays(forecast, 0, 0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    draw_forecast(axes[0], hist_times, temp_context[-TAIL:], future_times, median, low, high, temp_future, "°C")
    axes[0].set_title("Shanghai air temperature, day-of-week codes target-encoded per series")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].step(bundle["times"][shown], clock["day_of_week"][shown], where="post", label="day of week (Monday=0 … Sunday=6)")
    axes[1].axvline(future_times[0], color="0.5", lw=0.8, ls=":")
    axes[1].set_ylabel("code")
    axes[1].set_xlabel("UTC time")
    axes[1].set_yticks(range(7))
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Categorical known covariate — integer day-of-week codes, including the horizon")
    path = FIG_DIR / "06_categorical_covariate.png"
    finish(fig, path)
    summary["examples"]["categorical"] = note_from(forecast, elapsed, list(forecast.quantiles.shape))
    summary["examples"]["categorical"]["figure"] = str(path.relative_to(ROOT.parent))
    print(f"6 categorical {tuple(forecast.quantiles.shape)} {elapsed:.1f}s", flush=True)

    (DATA_DIR / "forecast_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote", DATA_DIR / "forecast_summary.json", flush=True)


if __name__ == "__main__":
    main()
