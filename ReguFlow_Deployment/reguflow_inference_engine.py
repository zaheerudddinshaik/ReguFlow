from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

import train_reguflow as base
from train_reguflow_coupled import CoupledReguFlowModel


ROOT = Path(__file__).resolve().parent

HORIZONS = (1, 6, 12, 24)

POLLUTANTS = (
    "PM2.5", "PM10", "NO2",
    "SO2", "CO", "Ozone"
)

MODEL_CONFIGS = {
    "Original ReguFlow": (
        "reguflow_models",
        base.ReguFlowModel
    ),
    "Coupled ReguFlow": (
        "reguflow_coupled_models",
        CoupledReguFlowModel
    )
}


@lru_cache(maxsize=8)
def load_model(model_name, horizon):

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if horizon not in HORIZONS:
        raise ValueError(
            f"Unsupported horizon: {horizon}"
        )

    folder, model_class = MODEL_CONFIGS[model_name]

    checkpoint_path = (
        ROOT / folder / f"{horizon}h" /
        "reguflow_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["horizon_hours"] != horizon:
        raise ValueError(
            "Checkpoint horizon mismatch."
        )

    model = model_class()

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    return model, checkpoint


@lru_cache(maxsize=4)
def load_test_data(horizon):

    path = (
        ROOT / "reguflow_scaled" /
        f"HB_Colony_Vijayawada_{horizon}h_"
        "reguflow_test_scaled.csv"
    )

    df = pd.read_csv(path)

    df["Forecast_Issue_Time"] = pd.to_datetime(
        df["Forecast_Issue_Time"]
    )

    df["Target_Time"] = pd.to_datetime(
        df["Target_Time"]
    )

    if df["Forecast_Issue_Time"].duplicated().any():
        raise ValueError(
            f"Duplicate forecast issue times: {horizon}h"
        )

    return df.set_index(
        "Forecast_Issue_Time",
        drop=False
    )


def available_issue_times():

    common = None

    for horizon in HORIZONS:

        times = set(
            load_test_data(horizon).index
        )

        if common is None:
            common = times
        else:
            common &= times

    return sorted(common)


def build_input(row, checkpoint):

    temporal_steps = []
    regulatory_steps = []

    for lag in range(47, -1, -1):

        suffix = f"__lag_{lag:02d}"

        temporal_columns = [
            name + suffix
            for name in checkpoint["temporal_features"]
        ]

        regulatory_columns = [
            name + suffix
            for name in checkpoint["regulatory_features"]
        ]

        temporal_steps.append(
            row[temporal_columns].to_numpy(
                dtype=np.float32
            )
        )

        regulatory_steps.append(
            row[regulatory_columns].to_numpy(
                dtype=np.float32
            )
        )

    temporal = np.stack(
        temporal_steps,
        axis=0
    )[None, :, :]

    regulatory = np.stack(
        regulatory_steps,
        axis=0
    )[None, :, :]

    if temporal.shape != (1, 48, 27):
        raise ValueError(
            f"Invalid temporal shape: {temporal.shape}"
        )

    if regulatory.shape != (1, 48, 98):
        raise ValueError(
            f"Invalid regulatory shape: {regulatory.shape}"
        )

    if (
        not np.isfinite(temporal).all()
        or not np.isfinite(regulatory).all()
    ):
        raise ValueError(
            "Non-finite inference features."
        )

    return (
        torch.from_numpy(temporal),
        torch.from_numpy(regulatory)
    )


def forecast(model_name, horizon, issue_time):

    model, checkpoint = load_model(
        model_name,
        horizon
    )

    df = load_test_data(horizon)

    issue_time = pd.Timestamp(issue_time)

    if issue_time not in df.index:
        raise ValueError(
            f"Issue time unavailable for {horizon}h: "
            f"{issue_time}"
        )

    row = df.loc[issue_time]

    temporal, regulatory = build_input(
        row,
        checkpoint
    )

    with torch.inference_mode():

        standardized = model(
            temporal,
            regulatory
        ).cpu().numpy()[0]

    mean = np.asarray(
        checkpoint["target_mean"],
        dtype=np.float32
    )

    std = np.asarray(
        checkpoint["target_std"],
        dtype=np.float32
    )

    prediction = standardized * std + mean

    target_time = pd.Timestamp(
        row["Target_Time"]
    )

    if target_time != issue_time + pd.Timedelta(
        hours=horizon
    ):
        raise ValueError(
            "Forecast target timestamp mismatch."
        )

    result = {
        "Model": model_name,
        "Horizon_hours": horizon,
        "Forecast_Issue_Time": issue_time,
        "Target_Time": target_time
    }

    for pollutant, value in zip(
        POLLUTANTS,
        prediction
    ):
        result[pollutant] = float(value)

    return result


def forecast_all(issue_time, model_names=None):

    if model_names is None:
        model_names = list(MODEL_CONFIGS)

    rows = []

    for model_name in model_names:
        for horizon in HORIZONS:

            rows.append(
                forecast(
                    model_name,
                    horizon,
                    issue_time
                )
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":

    times = available_issue_times()

    if not times:
        raise RuntimeError(
            "No common historical forecast issue times."
        )

    print("=" * 78)
    print("AQISTRATA - REGUFLOW INFERENCE ENGINE")
    print("=" * 78)

    print("Available issue times:", len(times))
    print("First issue time:", times[0])
    print("Last issue time:", times[-1])

    print("\nGenerating forecasts...\n")

    results = forecast_all(times[0])

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    assert len(results) == 8

    assert results[
        list(POLLUTANTS)
    ].notna().all().all()

    print("\nRESULT: INFERENCE ENGINE SMOKE TEST PASSED")
