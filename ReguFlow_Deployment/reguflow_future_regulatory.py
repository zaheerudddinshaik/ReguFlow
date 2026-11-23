import re

import numpy as np
import pandas as pd

from reguflow_live_regulatory import build_live_aqi_features
from regulatory_state_live_features import (
    build_regulatory_state_features,
)

HORIZONS = [1, 6, 12, 24]


def forecast_future_regulatory(history, predictions):
    """
    Reconstruct future regulatory-state scenarios from
    direct multi-horizon pollutant forecasts.

    Intermediate hours are linearly interpolated.
    Results are scenario estimates, not independent
    hourly model predictions.
    """

    history = history.copy()
    predictions = predictions.copy()

    history["Timestamp"] = pd.to_datetime(
        history["Timestamp"]
    )

    from reguflow_live_regulatory import POLLUTANT_WINDOWS

    pollutants = list(POLLUTANT_WINDOWS)

    missing = [
        name for name in pollutants
        if name not in history.columns
        or name + "__was_missing" not in history.columns
    ]

    if missing:
        raise ValueError(
            "Missing historical pollutant columns: "
            + repr(missing)
        )

    model_pollutants = [
        "PM2.5", "PM10", "NO2",
        "SO2", "CO", "Ozone",
    ]

    if not all(c in predictions.columns for c in model_pollutants):
        raise ValueError(
            "Forecast output is missing pollutants."
        )

    latest_time = history["Timestamp"].max()

    results = []

    for model_name, group in predictions.groupby("Model"):

        group = group.set_index("Horizon_hours")

        if group.index.has_duplicates:
            raise ValueError(
                f"Duplicate forecast horizons: {model_name}"
            )

        if not all(h in group.index for h in HORIZONS):
            raise ValueError(
                f"Missing forecast horizons for {model_name}"
            )

        future = pd.DataFrame({
            "Timestamp": [
                latest_time + pd.Timedelta(hours=h)
                for h in range(1, 25)
            ]
        })

        x = np.array([0, 1, 6, 12, 24])

        for model_col in model_pollutants:

            matches = [
                col for col in pollutants
                if re.sub(
                    r"[^a-z0-9.]",
                    "",
                    col.lower().split(" (")[0]
                ) == model_col.lower()
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Cannot uniquely map {model_col}: {matches}"
                )

            history_col = matches[0]

            y = np.array([
                float(history[history_col].iloc[-1]),
                *[
                    float(group.loc[h, model_col])
                    for h in HORIZONS
                ],
            ])

            future[history_col] = np.interp(
                np.arange(1, 25),
                x,
                y,
            )

            future[
                history_col + "__was_missing"
            ] = 0

        # Regulatory builders need a contiguous timeline.
        combined = pd.concat(
            [history, future],
            ignore_index=True,
        )

        combined = combined.sort_values(
            "Timestamp"
        ).reset_index(drop=True)

        # Recalculate future AQI from forecast
        # concentrations and historical context.
        combined = build_live_aqi_features(combined)

        combined = build_regulatory_state_features(
            combined
        )

        selected = combined[
            combined["Timestamp"].isin(
                future["Timestamp"]
            )
        ].copy()

        selected["Horizon_hours"] = (
            selected["Timestamp"] - latest_time
        ).dt.total_seconds() / 3600

        selected["Model"] = model_name

        selected["Forecast_Method"] = (
            "Direct horizon forecasts + "
            "linear intermediate-hour scenario"
        )

        results.append(selected)

    return pd.concat(
        results,
        ignore_index=True,
    )
