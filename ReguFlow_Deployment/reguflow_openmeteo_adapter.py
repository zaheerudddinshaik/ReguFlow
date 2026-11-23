from pathlib import Path

import requests
import pandas as pd
import numpy as np

LATITUDE = 16.5062
LONGITUDE = 80.6480
TIMEZONE = "Asia/Kolkata"

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_VARIABLES = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
    "ozone",
]

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "shortwave_radiation",
    "surface_pressure",
]


def fetch_json(url, params):
    response = requests.get(
        url,
        params=params,
        timeout=45,
    )
    response.raise_for_status()
    return response.json()



def normalize_model_columns(df):
    """
    Align Open-Meteo column names with the exact
    names used by the trained ReguFlow models.
    """

    import torch

    checkpoint = torch.load(
        Path(__file__).resolve().parent
        / "reguflow_models"
        / "1h"
        / "reguflow_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    expected = list(checkpoint["temporal_features"][:14])

    rename = {}

    for name in expected:

        if name in df.columns:
            continue

        prefix = name.split(" (")[0]

        matches = [
            column for column in df.columns
            if column.split(" (")[0] == prefix
            and not column.endswith("__was_missing")
        ]

        if len(matches) == 1:
            old = matches[0]

            rename[old] = name

            old_flag = old + "__was_missing"
            new_flag = name + "__was_missing"

            if old_flag in df.columns:
                rename[old_flag] = new_flag

    df = df.rename(columns=rename)

    missing = [
        name for name in expected
        if name not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing model input columns: "
            + repr(missing)
        )

    return df

def fetch_current_inputs():

    now = pd.Timestamp.now(tz=TIMEZONE)
    today = now.date()
    start = today - pd.Timedelta(days=5)

    air = fetch_json(
        AIR_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(AIR_VARIABLES),
            "past_days": 6,
            "forecast_days": 1,
            "timezone": TIMEZONE,
        },
    )

    weather = fetch_json(
        WEATHER_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": str(start),
            "end_date": str(today),
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": TIMEZONE,
        },
    )

    air_df = pd.DataFrame(air["hourly"])
    weather_df = pd.DataFrame(weather["hourly"])

    air_df = air_df.rename(
        columns={
            "time": "Timestamp",
            "pm2_5": "PM2.5 (?g/m?)",
            "pm10": "PM10 (?g/m?)",
            "nitrogen_dioxide": "NO2 (?g/m?)",
            "sulphur_dioxide": "SO2 (?g/m?)",
            "carbon_monoxide": "CO (mg/m?)",
            "ozone": "Ozone (?g/m?)",
        }
    )

    weather_df = weather_df.rename(
        columns={
            "time": "Timestamp",
            "temperature_2m": "AT (?C)",
            "relative_humidity_2m": "RH (%)",
            "wind_speed_10m": "WS (m/s)",
            "wind_direction_10m": "WD (deg)",
            "precipitation": "RF (mm)",
            "shortwave_radiation": "SR (W/mt2)",
            "surface_pressure": "Surface_Pressure_hPa",
        }
    )

    air_df["Timestamp"] = pd.to_datetime(
        air_df["Timestamp"]
    )

    weather_df["Timestamp"] = pd.to_datetime(
        weather_df["Timestamp"]
    )

    # Never use future API forecasts as observed inputs.
    current_hour = now.tz_localize(None).floor("h")

    air_df = air_df[
        air_df["Timestamp"] <= current_hour
    ]

    weather_df = weather_df[
        weather_df["Timestamp"] <= current_hour
    ]

    # Convert carbon monoxide from ?g/m? to mg/m?.
    air_df["CO (mg/m?)"] = (
        pd.to_numeric(
            air_df["CO (mg/m?)"],
            errors="coerce",
        ) / 1000.0
    )

    # Convert wind speed from km/h to m/s.
    weather_df["WS (m/s)"] = (
        pd.to_numeric(
            weather_df["WS (m/s)"],
            errors="coerce",
        ) / 3.6
    )

    # Convert pressure from hPa to mmHg.
    weather_df["BP (mmHg)"] = (
        pd.to_numeric(
            weather_df["Surface_Pressure_hPa"],
            errors="coerce",
        ) * 0.750061683
    )

    weather_df = weather_df.drop(
        columns=["Surface_Pressure_hPa"]
    )

    df = pd.merge(
        air_df,
        weather_df,
        on="Timestamp",
        how="inner",
        validate="one_to_one",
    )

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    # Total rainfall over a trailing 24-hour window.
    df["TOT-RF (mm)"] = (
        df["RF (mm)"]
        .rolling(24, min_periods=24)
        .sum()
    )

    for pollutant in [
        "PM2.5 (?g/m?)",
        "PM10 (?g/m?)",
        "NO2 (?g/m?)",
        "SO2 (?g/m?)",
        "CO (mg/m?)",
        "Ozone (?g/m?)",
    ]:
        df[pollutant + "__was_missing"] = (
            df[pollutant].isna().astype(int)
        )

    return normalize_model_columns(df)


if __name__ == "__main__":

    print("=" * 70)
    print("AQISTRATA - OPEN-METEO DATA ADAPTER")
    print("=" * 70)

    df = fetch_current_inputs()

    print("Rows:", len(df))
    print("First timestamp:", df["Timestamp"].min())
    print("Last timestamp:", df["Timestamp"].max())

    print("\nColumns:")
    for column in df.columns:
        print(" ", column)

    print("\nLatest 48-hour missing values:")
    print(df.tail(48).isna().sum().to_string())

    if len(df) >= 48:
        intervals = df.tail(48)["Timestamp"].diff().dropna()
        print(
            "\n48-hour continuity:",
            "PASS" if intervals.eq(
                pd.Timedelta(hours=1)
            ).all() else "FAIL",
        )

    print("\nData source: Open-Meteo model estimates")
    print("Not CPCB station observations.")
