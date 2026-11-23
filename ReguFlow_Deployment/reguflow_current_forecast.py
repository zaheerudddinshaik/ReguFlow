
import numpy as np
import pandas as pd
import torch

from reguflow_inference_engine import (
    load_model,
    MODEL_CONFIGS,
    HORIZONS,
    POLLUTANTS,
)

from reguflow_live_preprocessing import (
    prepare_live_model_inputs,
)


def forecast_current(hourly_df, model_names=None):

    if model_names is None:
        model_names = list(MODEL_CONFIGS)

    rows = []

    for horizon in HORIZONS:

        temporal, regulatory, issue_time = (
            prepare_live_model_inputs(
                hourly_df,
                horizon_hours=horizon,
            )
        )

        for model_name in model_names:

            model, checkpoint = load_model(
                model_name,
                horizon,
            )

            with torch.inference_mode():
                output = model(
                    temporal,
                    regulatory,
                )

            if isinstance(output, (tuple, list)):
                output = output[0]

            prediction = (
                output.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            if prediction.shape != (len(POLLUTANTS),):
                raise ValueError(
                    "Unexpected model output shape: "
                    + repr(prediction.shape)
                )

            target_mean = np.asarray(
                checkpoint["target_mean"],
                dtype=float,
            )

            target_std = np.asarray(
                checkpoint["target_std"],
                dtype=float,
            )

            if (
                target_mean.shape != prediction.shape
                or target_std.shape != prediction.shape
            ):
                raise ValueError(
                    "Checkpoint target statistics mismatch."
                )

            prediction = (
                prediction * target_std + target_mean
            )

            if not np.isfinite(prediction).all():
                raise ValueError(
                    "Non-finite forecast values."
                )

            row = {
                "Model": model_name,
                "Horizon_hours": horizon,
                "Forecast_Issue_Time": issue_time,
                "Target_Time": (
                    issue_time
                    + pd.Timedelta(hours=horizon)
                ),
            }

            row.update(
                dict(zip(POLLUTANTS, prediction))
            )

            rows.append(row)

    return pd.DataFrame(rows)
