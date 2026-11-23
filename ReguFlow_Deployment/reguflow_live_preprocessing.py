
"""
Live preprocessing for the four trained forecasting horizons.

Input:
    Hourly observations with historical regulatory features.

Output:
    One model-ready temporal tensor and regulatory tensor.

The function uses saved training medians and scalers.
It never fits preprocessing artifacts on live data.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parent
HORIZONS = (1, 6, 12, 24)


def prepare_live_model_inputs(hourly_df, horizon_hours=1):
    """
    Prepare the latest 48-hour window for one saved model.

    Returns:
        temporal_tensor: (1, 48, 27)
        regulatory_tensor: (1, 48, 98)
        issue_time: latest observation timestamp
    """

    if horizon_hours not in HORIZONS:
        raise ValueError(
            f"Unsupported horizon: {horizon_hours}"
        )

    if not isinstance(hourly_df, pd.DataFrame):
        raise TypeError("hourly_df must be a DataFrame")

    checkpoint_path = (
        PROJECT_DIR
        / "reguflow_models"
        / f"{horizon_hours}h"
        / "reguflow_best.pt"
    )

    scaler_path = (
        PROJECT_DIR
        / "reguflow_scaled"
        / (
            f"HB_Colony_Vijayawada_{horizon_hours}h"
            "_reguflow_scaler.joblib"
        )
    )

    median_path = (
        PROJECT_DIR
        / "reguflow_imputed"
        / (
            f"HB_Colony_Vijayawada_{horizon_hours}h"
            "_reguflow_training_feature_medians.csv"
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    scaler = joblib.load(scaler_path)
    median_df = pd.read_csv(median_path)

    temporal_features = list(
        checkpoint["temporal_features"]
    )

    regulatory_features = list(
        checkpoint["regulatory_features"]
    )

    numeric_regulatory = [
        c for c in regulatory_features
        if not c.endswith("__is_missing")
    ]

    temporal_numeric = [
        c for c in temporal_features
        if not c.endswith("__was_missing")
    ]

    numeric_features = (
        temporal_numeric + numeric_regulatory
    )

    if (
        len(temporal_features) != 27
        or len(regulatory_features) != 98
        or len(numeric_features) != 63
    ):
        raise ValueError(
            "Unexpected checkpoint feature configuration"
        )

    df = hourly_df.copy()

    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column missing")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="raise",
    )

    if df["Timestamp"].isna().any():
        raise ValueError("Invalid timestamps")

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps")

    if len(df) < 48:
        raise ValueError(
            "At least 48 hourly observations required"
        )

    window = df.tail(48).copy()

    intervals = window["Timestamp"].diff().dropna()

    if not intervals.eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "Latest 48 observations are not hourly-contiguous"
        )

    issue_time = window["Timestamp"].iloc[-1]

    missing_numeric = [
        c for c in numeric_features
        if c not in window.columns
    ]

    if missing_numeric:
        raise ValueError(
            "Missing numeric features: "
            + repr(missing_numeric)
        )

    # Construct feature values and missingness flags.
    # Preserve supplied source-missing flags when available.
    for feature in temporal_numeric:

        flag = feature + "__was_missing"

        values = pd.to_numeric(
            window[feature],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        window[feature] = values

        if flag in temporal_features:

            if flag in window.columns:
                supplied = pd.to_numeric(
                    window[flag],
                    errors="coerce",
                )

                if (
                    supplied.isna().any()
                    or not supplied.isin([0, 1]).all()
                ):
                    raise ValueError(
                        f"Invalid source missingness flag: {flag}"
                    )

                window[flag] = supplied.astype(float)

            else:
                window[flag] = (
                    values.isna().astype(float)
                )

    for feature in numeric_regulatory:

        values = pd.to_numeric(
            window[feature],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        window[feature] = values

        window[feature + "__is_missing"] = (
            values.isna().astype(float)
        )

    # Lag 00 is newest; lag 47 is oldest.
    window = window.iloc[::-1].reset_index(drop=True)

    scaler_names = list(scaler.feature_names_in_)

    if len(scaler_names) != 3024:
        raise ValueError(
            "Unexpected saved scaler feature count"
        )

    expected_numeric_names = [
        f"{feature}__lag_{lag:02d}"
        for lag in range(48)
        for feature in numeric_features
    ]

    if scaler_names != expected_numeric_names:
        raise ValueError(
            "Saved scaler ordering differs from checkpoint"
        )

    median_names = median_df.iloc[:, 0].astype(str).tolist()

    if median_names != scaler_names:
        raise ValueError(
            "Saved training medians are misaligned"
        )

    median_values = pd.to_numeric(
        median_df["training_median"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(median_values).all():
        raise ValueError("Non-finite training medians")

    numeric_values = np.array(
        [
            window.loc[lag, feature]
            for lag in range(48)
            for feature in numeric_features
        ],
        dtype=float,
    )

    missing = ~np.isfinite(numeric_values)

    numeric_values[missing] = median_values[missing]

    numeric_frame = pd.DataFrame(
        [numeric_values],
        columns=scaler_names,
    )

    scaled_values = scaler.transform(
        numeric_frame
    )[0]

    scaled_lookup = dict(
        zip(scaler_names, scaled_values)
    )

    temporal_rows = []
    regulatory_rows = []

    for lag in range(48):

        temporal_row = []

        for feature in temporal_features:

            if feature in temporal_numeric:
                value = scaled_lookup[
                    f"{feature}__lag_{lag:02d}"
                ]
            else:
                value = window.loc[lag, feature]

            temporal_row.append(float(value))

        regulatory_row = []

        for feature in regulatory_features:

            if feature in numeric_regulatory:
                value = scaled_lookup[
                    f"{feature}__lag_{lag:02d}"
                ]
            else:
                value = window.loc[lag, feature]

            regulatory_row.append(float(value))

        temporal_rows.append(temporal_row)
        regulatory_rows.append(regulatory_row)

    # Chronological model sequence: oldest to newest.
    # Lag 47 comes first; lag 00 comes last.
    temporal_array = np.asarray(
        temporal_rows[::-1],
        dtype=np.float32,
    )

    regulatory_array = np.asarray(
        regulatory_rows[::-1],
        dtype=np.float32,
    )

    if temporal_array.shape != (48, 27):
        raise ValueError("Invalid temporal shape")

    if regulatory_array.shape != (48, 98):
        raise ValueError("Invalid regulatory shape")

    if (
        not np.isfinite(temporal_array).all()
        or not np.isfinite(regulatory_array).all()
    ):
        raise ValueError(
            "Non-finite values in prepared model input"
        )

    return (
        torch.from_numpy(
            temporal_array[None, :, :]
        ),
        torch.from_numpy(
            regulatory_array[None, :, :]
        ),
        issue_time,
    )
