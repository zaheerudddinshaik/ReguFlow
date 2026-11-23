
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. FILE CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR /
    "HB_Colony_Vijayawada_regulatory_features_candidate.csv"
)

OUTPUT_FILE = (
    BASE_DIR /
    "HB_Colony_Vijayawada_reguflow_flux_features_candidate.csv"
)

BOUNDARY_EPSILON = 1e-6
COMPETITION_EPSILON = 1.0

AQI_BOUNDARIES = np.array(
    [50, 100, 200, 300, 400, 500],
    dtype=float
)


# ---------------------------------------------------------
# 2. POLLUTANT SUB-INDEX COLUMNS
# ---------------------------------------------------------

POLLUTANTS = {
    "PM2.5": "PM2.5 (µg/m³)__subindex",
    "PM10": "PM10 (µg/m³)__subindex",
    "NO2": "NO2 (µg/m³)__subindex",
    "SO2": "SO2 (µg/m³)__subindex",
    "CO": "CO (mg/m³)__subindex",
    "Ozone": "Ozone (µg/m³)__subindex",
}

TIMESTAMP_CANDIDATES = [
    "Timestamp",
    "timestamp",
    "Datetime",
    "datetime",
    "Date",
    "date",
    "From Date",
]


# ---------------------------------------------------------
# 3. HELPER: FIND NEAREST AQI BOUNDARY
# ---------------------------------------------------------

def find_nearest_boundary(values):
    """
    Finds the nearest AQI boundary for each pollutant
    sub-index value.

    Missing values remain NaN.
    """

    values = pd.to_numeric(
        values,
        errors="coerce"
    )

    array = values.to_numpy(dtype=float)

    result = np.full(
        len(array),
        np.nan,
        dtype=float
    )

    valid = np.isfinite(array)

    if valid.any():
        distances = np.abs(
            array[valid, None] - AQI_BOUNDARIES[None, :]
        )

        nearest_indices = np.argmin(
            distances,
            axis=1
        )

        result[valid] = AQI_BOUNDARIES[
            nearest_indices
        ]

    return pd.Series(
        result,
        index=values.index
    )


# ---------------------------------------------------------
# 4. LOAD AND VALIDATE DATA
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("REGUFLOW FLUX FEATURE EXTRACTION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nInput file loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # Find timestamp column
    timestamp_column = next(
        (
            column
            for column in TIMESTAMP_CANDIDATES
            if column in df.columns
        ),
        None
    )

    if timestamp_column is None:
        raise ValueError(
            "No recognized timestamp column was found."
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )

    if df[timestamp_column].isna().any():
        raise ValueError(
            "Timestamp column contains invalid values."
        )

    if df[timestamp_column].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    if not df[timestamp_column].is_monotonic_increasing:
        raise ValueError(
            "Data is not chronologically sorted."
        )

    # Validate pollutant columns
    missing_columns = [
        column
        for column in POLLUTANTS.values()
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing pollutant sub-index columns: "
            f"{missing_columns}"
        )

    # Actual elapsed time in hours
    elapsed_hours = (
        df[timestamp_column]
        .diff()
        .dt.total_seconds()
        / 3600.0
    )

    invalid_intervals = (
        elapsed_hours.notna()
        & (elapsed_hours <= 0)
    )

    if invalid_intervals.any():
        raise ValueError(
            "Non-positive timestamp intervals detected."
        )

    print("Timestamp validation: PASS")
    print(
        "Elapsed interval counts (hours):",
        elapsed_hours.value_counts(dropna=True)
        .sort_index()
        .to_dict()
    )


    # -----------------------------------------------------
    # 5. POLLUTANT BOUNDARY VELOCITY AND FLUX
    # -----------------------------------------------------

    print("\nCalculating pollutant boundary velocities and fluxes...")

    for pollutant, column in POLLUTANTS.items():

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Velocity: change in sub-index per elapsed hour
        velocity = values.diff() / elapsed_hours

        # Nearest regulatory AQI boundary
        nearest_boundary = find_nearest_boundary(
            values
        )

        # Guide convention:
        # q = nearest boundary - current sub-index
        signed_distance = (
            nearest_boundary - values
        )

        # Boundary flux:
        # FB = velocity * sign(q) / (abs(q) + epsilon)
        boundary_flux = (
            velocity
            * np.sign(signed_distance)
            / (np.abs(signed_distance) + BOUNDARY_EPSILON)
        )

        # Save new features
        df[f"{pollutant}_Boundary_Velocity"] = velocity

        df[f"{pollutant}_ReguFlow_Signed_Distance"] = (
            signed_distance
        )

        df[f"{pollutant}_Boundary_Flux"] = boundary_flux

        print(
            f"  {pollutant}: velocity and boundary flux calculated"
        )


    
    # -----------------------------------------------------
    # 6. TOP-TWO DOMINANCE GAP
    # -----------------------------------------------------

    print("\nCalculating top-two dominance gap...")

    subindex_columns = list(POLLUTANTS.values())

    subindices = df[subindex_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    values = subindices.to_numpy(dtype=float)

    valid_count = np.isfinite(values).sum(axis=1)

    # Temporarily replace missing values with -infinity.
    # This prevents missing values from becoming top values.
    sortable_values = np.where(
        np.isfinite(values),
        values,
        -np.inf
    )

    dominance_gap = np.full(
        len(values),
        np.nan,
        dtype=float
    )

    # At least two valid pollutant sub-indices are required.
    valid_rows = valid_count >= 2

    if valid_rows.any():

        valid_values = sortable_values[valid_rows]

        sorted_valid_values = np.sort(
            valid_values,
            axis=1
        )

        top1 = sorted_valid_values[:, -1]
        top2 = sorted_valid_values[:, -2]

        dominance_gap[valid_rows] = top1 - top2

    gap_series = pd.Series(
        dominance_gap,
        index=df.index
    )

    df["ReguFlow_Top2_Dominance_Gap"] = gap_series



    
    # -----------------------------------------------------
    # 7. NORMALIZED COMPETITION FLUX
    # -----------------------------------------------------

    print("Calculating normalized competition flux...")

    # Compare the same available pollutants at t and t-1.
    # At least two common valid pollutants are required.

    competition_flux_values = np.full(
        len(values),
        np.nan,
        dtype=float
    )

    comparable_transition_count = 0

    for i in range(1, len(values)):

        current_values = values[i]
        previous_values = values[i - 1]

        # Identify pollutants valid at BOTH timestamps
        common_mask = (
            np.isfinite(current_values)
            & np.isfinite(previous_values)
        )

        # Need at least two common pollutants
        if np.sum(common_mask) < 2:
            continue

        current_common = np.sort(
            current_values[common_mask]
        )

        previous_common = np.sort(
            previous_values[common_mask]
        )

        # Calculate both gaps using exactly the same pollutants
        current_gap = (
            current_common[-1] - current_common[-2]
        )

        previous_gap = (
            previous_common[-1] - previous_common[-2]
        )

        # Normalized competition flux
        competition_flux_values[i] = (
            -(current_gap - previous_gap)
            / (current_gap + COMPETITION_EPSILON)
        )

        comparable_transition_count += 1

    df["ReguFlow_Competition_Flux"] = pd.Series(
        competition_flux_values,
        index=df.index
    )

    print(
        "Comparable pollutant transitions:",
        comparable_transition_count
    )



    # -----------------------------------------------------
    # 8. FEATURE VALIDATION
    # -----------------------------------------------------

    new_columns = []

    for pollutant in POLLUTANTS:

        new_columns.extend([
            f"{pollutant}_Boundary_Velocity",
            f"{pollutant}_ReguFlow_Signed_Distance",
            f"{pollutant}_Boundary_Flux",
        ])

    new_columns.extend([
        "ReguFlow_Top2_Dominance_Gap",
        "ReguFlow_Competition_Flux",
    ])

    print("\n" + "=" * 65)
    print("FEATURE VALIDATION")
    print("=" * 65)

    print("New ReguFlow feature count:", len(new_columns))

    for column in new_columns:

        missing = int(df[column].isna().sum())

        finite_values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(finite_values) > 0:
            minimum = finite_values.min()
            maximum = finite_values.max()
        else:
            minimum = np.nan
            maximum = np.nan

        print(
            f"{column}: "
            f"missing={missing}, "
            f"min={minimum:.6f}, "
            f"max={maximum:.6f}"
        )

    # Check for infinite values in newly generated features
    for column in new_columns:

        df[column] = df[column].replace(
            [np.inf, -np.inf],
            np.nan
        )


    # -----------------------------------------------------
    # 9. SAVE SEPARATE OUTPUT FILE
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 65)
    print("REGUFLOW FEATURE EXTRACTION COMPLETED")
    print("=" * 65)

    print("Output file:", OUTPUT_FILE.name)
    print("Total rows:", len(df))
    print("Original columns preserved:", True)
    print("New ReguFlow features:", len(new_columns))
    print("Total output columns:", len(df.columns))

    print("\nNew features created:")

    for column in new_columns:
        print(" -", column)

    print("\nSUCCESS: New candidate dataset saved.")


if __name__ == "__main__":
    main()
