
import numpy as np
import pandas as pd

from extract_regulatory_features import (
    POLLUTANTS,
    get_nearest_boundary,
    get_absolute_boundary_distance,
)

from extract_reguflow_flux_features import (
    find_nearest_boundary,
    BOUNDARY_EPSILON,
    COMPETITION_EPSILON,
)


def build_regulatory_state_features(aqi_df):
    """
    Generate regulatory-state features from an hourly
    dataframe containing historical-style AQI features.

    Does not impute, scale, or modify model artifacts.
    """

    df = aqi_df.copy()

    required = [
        "Timestamp",
        "AQI",
        "Dominant_Pollutant",
        "Dominance_Gap",
        *POLLUTANTS.values(),
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="raise",
    )

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps.")

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    elapsed = (
        df["Timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600.0)
    )

    if (elapsed.dropna() != 1).any():
        raise ValueError(
            "Hourly contiguous observations required."
        )

    for column in ["AQI", "Dominance_Gap", *POLLUTANTS.values()]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

    subindices = df[list(POLLUTANTS.values())]

    values = subindices.to_numpy(dtype=float)
    finite = np.isfinite(values)
    valid_count = finite.sum(axis=1)

    sortable = np.where(finite, values, -np.inf)
    sorted_values = np.sort(sortable, axis=1)

    # Historical merged-dataset conventions.
    df["Valid_Pollutant_Count"] = valid_count
    df["Valid_Subindex_Count"] = valid_count

    df["Second_Subindex"] = np.where(
        np.isfinite(df["AQI"].to_numpy())
        & (valid_count >= 2),
        sorted_values[:, -2],
        np.nan,
    )

    # AQI regulatory boundary geometry.
    df["AQI_Boundary_Distance"] = df["AQI"].apply(
        get_absolute_boundary_distance
    )

    df["Nearest_AQI_Boundary"] = df["AQI"].apply(
        get_nearest_boundary
    )

    for pollutant, column in POLLUTANTS.items():

        df[f"{pollutant}_Boundary_Distance"] = (
            df[column].apply(
                get_absolute_boundary_distance
            )
        )

    # Co-elevation.
    df["CoElevated_Count_Above_50"] = (
        subindices.gt(50).sum(axis=1)
    )

    df["CoElevated_Count_Above_100"] = (
        subindices.gt(100).sum(axis=1)
    )

    # Dominance indicators.
    for pollutant in POLLUTANTS:
        df[f"Dominant_{pollutant}"] = (
            df["Dominant_Pollutant"]
            .eq(pollutant)
            .astype("int8")
        )

    # AQI boundary velocity.
    aqi_change = df["AQI"].diff()

    df["AQI_Boundary_Velocity"] = (
        aqi_change / elapsed
    ).where(
        elapsed.gt(0) & aqi_change.notna()
    )

    # Twelve-hour dominance stability.
    dominant = (
        df["Dominant_Pollutant"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    stability_parts = []

    for pollutant in POLLUTANTS:
        matches = dominant.eq(pollutant).astype("float64")
        matches = matches.mask(dominant.isna())
        stability_parts.append(
            matches.rename(pollutant)
        )

    match_matrix = pd.concat(
        stability_parts,
        axis=1,
    )

    valid_window = (
        dominant.notna()
        .rolling(12, min_periods=12)
        .sum()
        .eq(12)
    )

    stability_values = match_matrix.rolling(
        12,
        min_periods=12,
    ).mean()

    current_match = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for pollutant in POLLUTANTS:
        mask = dominant.eq(pollutant).fillna(False)

        current_match.loc[mask] = (
            stability_values.loc[mask, pollutant]
        )

    df["Dominance_Stability_12h"] = (
        current_match.where(valid_window)
    )

    # Pollutant boundary velocity and flux.
    for pollutant, column in POLLUTANTS.items():

        pollutant_values = df[column]

        velocity = (
            pollutant_values.diff() / elapsed
        )

        nearest = find_nearest_boundary(
            pollutant_values
        )

        signed_distance = (
            nearest - pollutant_values
        )

        boundary_flux = (
            velocity
            * np.sign(signed_distance)
            / (
                np.abs(signed_distance)
                + BOUNDARY_EPSILON
            )
        )

        df[f"{pollutant}_Boundary_Velocity"] = velocity

        df[f"{pollutant}_ReguFlow_Signed_Distance"] = (
            signed_distance
        )

        df[f"{pollutant}_Boundary_Flux"] = boundary_flux

    # Top-two dominance gap.
    top2_gap = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    valid_rows = valid_count >= 2

    top2_gap[valid_rows] = (
        sorted_values[valid_rows, -1]
        - sorted_values[valid_rows, -2]
    )

    df["ReguFlow_Top2_Dominance_Gap"] = top2_gap

    # Competition flux using common valid pollutants.
    competition_flux = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    for i in range(1, len(df)):

        common = finite[i] & finite[i - 1]

        if common.sum() < 2:
            continue

        current = np.sort(values[i, common])
        previous = np.sort(values[i - 1, common])

        current_gap = current[-1] - current[-2]
        previous_gap = previous[-1] - previous[-2]

        competition_flux[i] = (
            -(current_gap - previous_gap)
            / (current_gap + COMPETITION_EPSILON)
        )

    df["ReguFlow_Competition_Flux"] = competition_flux

    return df
