"""
AQISTRATA live regulatory preprocessing.

Reuses the project's existing rolling-average,
subindex and AQI calculations.

Input: hourly environmental DataFrame.
Output: DataFrame containing calculated AQI features.

This module does not modify historical project files.
"""

import numpy as np
import pandas as pd

from cpcb_averages import (
    POLLUTANT_WINDOWS,
    validate_missing_flags,
    generate_rolling_averages,
)
from cpcb_subindex import calculate_subindex
from calculate_overall_aqi import calculate_row_aqi


def build_live_aqi_features(hourly_df):

    df = hourly_df.copy()

    if "Timestamp" not in df.columns:
        raise ValueError("Missing Timestamp column.")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    if df["Timestamp"].isna().any():
        raise ValueError("Invalid timestamps.")

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps.")

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    if len(df) < 24:
        raise ValueError(
            "At least 24 hourly rows are required."
        )

    intervals = (
        df["Timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(3600)
    )

    if not intervals.eq(1).all():
        raise ValueError(
            "Input must contain contiguous hourly timestamps."
        )

    required = []

    for pollutant in POLLUTANT_WINDOWS:
        required.extend([
            pollutant,
            f"{pollutant}__was_missing",
        ])

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = validate_missing_flags(df)

    for pollutant in POLLUTANT_WINDOWS:

        values = pd.to_numeric(
            df[pollutant],
            errors="coerce",
        )

        invalid = (
            df[f"{pollutant}__was_missing"].eq(0)
            & ~np.isfinite(values)
        )

        if invalid.any():
            raise ValueError(
                f"Non-finite unflagged values: {pollutant}"
            )

        df[pollutant] = values

    df = generate_rolling_averages(df)

    subindex_columns = []

    for pollutant, window in POLLUTANT_WINDOWS.items():

        avg_column = f"{pollutant}__{window}h_avg"
        subindex_column = f"{pollutant}__subindex"

        df[subindex_column] = df[
            avg_column
        ].apply(
            lambda value, name=pollutant:
                calculate_subindex(name, value)
        )

        subindex_columns.append(subindex_column)

    aqi_features = df.apply(
        calculate_row_aqi,
        axis=1,
    )

    for column in aqi_features.columns:
        df[column] = aqi_features[column]

    return df


if __name__ == "__main__":
    print(
        "Live regulatory preprocessing module loaded."
    )
