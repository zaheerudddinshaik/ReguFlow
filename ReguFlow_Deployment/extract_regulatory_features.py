
import pandas as pd
from pathlib import Path
from itertools import combinations

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "HB_Colony_Vijayawada_aqi_candidate.csv"
OUTPUT_FILE = BASE_DIR / "HB_Colony_Vijayawada_regulatory_features_candidate.csv"


# --------------------------------------------------
# 2. POLLUTANT CONFIGURATION
# --------------------------------------------------

POLLUTANTS = {
    "PM2.5": "PM2.5 (µg/m³)__subindex",
    "PM10": "PM10 (µg/m³)__subindex",
    "NO2": "NO2 (µg/m³)__subindex",
    "SO2": "SO2 (µg/m³)__subindex",
    "CO": "CO (mg/m³)__subindex",
    "Ozone": "Ozone (µg/m³)__subindex",
}

# AQI category boundaries from the project configuration
BOUNDARIES = [50, 100, 200, 300, 400, 500]


# --------------------------------------------------
# 3. BOUNDARY DISTANCE FUNCTION
# --------------------------------------------------

def get_nearest_boundary(value):
    """
    Returns the nearest AQI category boundary.
    Returns None when the value is unavailable.
    """

    if pd.isna(value):
        return None

    return min(
        BOUNDARIES,
        key=lambda boundary: abs(value - boundary)
    )


def get_absolute_boundary_distance(value):
    """
    Absolute distance to the nearest AQI boundary.
    """

    if pd.isna(value):
        return None

    nearest = get_nearest_boundary(value)
    return abs(value - nearest)


def get_signed_boundary_distance(value):
    """
    Signed distance to the nearest AQI boundary.

    Positive: value is above the nearest boundary.
    Negative: value is below the nearest boundary.
    Zero: value is exactly on the boundary.
    """

    if pd.isna(value):
        return None

    nearest = get_nearest_boundary(value)
    return value - nearest


# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------

def main():

    print("Starting regulatory feature extraction...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows: {len(df):,}")

    # Check required columns
    required_columns = [
        "AQI",
        "Dominant_Pollutant",
        "Dominance_Gap",
        *POLLUTANTS.values(),
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert AQI and sub-indices to numeric
    df["AQI"] = pd.to_numeric(
        df["AQI"],
        errors="coerce"
    )

    df["Dominance_Gap"] = pd.to_numeric(
        df["Dominance_Gap"],
        errors="coerce"
    )

    for column in POLLUTANTS.values():
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------
    # 5. OVERALL AQI BOUNDARY FEATURES
    # --------------------------------------------------

    df["AQI_Boundary_Distance"] = df["AQI"].apply(
        get_absolute_boundary_distance
    )

    df["AQI_Signed_Boundary_Distance"] = df["AQI"].apply(
        get_signed_boundary_distance
    )

    df["Nearest_AQI_Boundary"] = df["AQI"].apply(
        get_nearest_boundary
    )

    # --------------------------------------------------
    # 6. PER-POLLUTANT BOUNDARY FEATURES
    # --------------------------------------------------

    for pollutant, column in POLLUTANTS.items():

        df[f"{pollutant}_Boundary_Distance"] = (
            df[column].apply(
                get_absolute_boundary_distance
            )
        )

        df[f"{pollutant}_Signed_Boundary_Distance"] = (
            df[column].apply(
                get_signed_boundary_distance
            )
        )

        df[f"{pollutant}_Nearest_Boundary"] = (
            df[column].apply(
                get_nearest_boundary
            )
        )

    # --------------------------------------------------
    # 7. CO-ELEVATION FEATURES
    # --------------------------------------------------

    subindices = df[list(POLLUTANTS.values())]

    # Count pollutants with sub-index above 50
    df["CoElevated_Count_Above_50"] = (
        subindices.gt(50)
        .sum(axis=1)
        .astype("int8")
    )

    # Count pollutants with sub-index above 100
    df["CoElevated_Count_Above_100"] = (
        subindices.gt(100)
        .sum(axis=1)
        .astype("int8")
    )

    # Count valid pollutant sub-indices
    df["Valid_Subindex_Count"] = (
        subindices.notna()
        .sum(axis=1)
        .astype("int8")
    )

    # --------------------------------------------------
    # 8. DOMINANT POLLUTANT FEATURES
    # --------------------------------------------------

    for pollutant in POLLUTANTS:

        df[f"Dominant_{pollutant}"] = (
            df["Dominant_Pollutant"]
            .eq(pollutant)
            .astype("int8")
        )

    # --------------------------------------------------
    # 9. PAIRWISE POLLUTANT DIFFERENCES
    # --------------------------------------------------

    # A positive value for pollutant_A_minus_pollutant_B
    # means A has a higher sub-index than B.
    # A negative value means B has a higher sub-index.
    # The feature is missing if either sub-index is missing.

    for (pollutant_a, column_a), (pollutant_b, column_b) in (
        combinations(POLLUTANTS.items(), 2)
    ):

        feature_name = (
            f"{pollutant_a}_minus_{pollutant_b}_Subindex"
        )

        df[feature_name] = (
            df[column_a] - df[column_b]
        )




    # --------------------------------------------------
    # 10. TEMPORAL REGUFLOW FEATURES
    # --------------------------------------------------

    # Preserve the original row order while calculating
    # temporal features in chronological order.
    original_index = df.index.copy()

    # Find the timestamp column used by this dataset.
    timestamp_candidates = [
        "Timestamp",
        "timestamp",
        "Datetime",
        "datetime",
        "DateTime",
        "date",
        "Date",
    ]

    timestamp_column = next(
        (
            column for column in timestamp_candidates
            if column in df.columns
        ),
        None,
    )

    if timestamp_column is None:
        raise ValueError(
            "No recognized timestamp column found. "
            "Inspect the CSV column names before calculating "
            "temporal features."
        )

    df["_original_order"] = range(len(df))

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    if df[timestamp_column].isna().any():
        raise ValueError(
            f"Invalid timestamps found in {timestamp_column}."
        )

    if df[timestamp_column].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found. Resolve them before "
            "calculating temporal features."
        )

    df = df.sort_values(timestamp_column).reset_index(drop=True)

    # 10A. AQI boundary velocity (AQI units per hour).
    elapsed_hours = (
        df[timestamp_column].diff().dt.total_seconds() / 3600
    )

    aqi_change = df["AQI"].diff()

    df["AQI_Boundary_Velocity"] = (
        aqi_change / elapsed_hours
    ).where(
        elapsed_hours.gt(0)
        & elapsed_hours.notna()
        & aqi_change.notna()
    )

    # 10B. Competition flux:
    # Change in the pairwise subindex difference per hour.
    for (pollutant_a, column_a), (pollutant_b, column_b) in (
        combinations(POLLUTANTS.items(), 2)
    ):
        gap = df[column_a] - df[column_b]
        gap_change = gap.diff()

        feature_name = (
            f"{pollutant_a}_vs_{pollutant_b}_Competition_Flux"
        )

        df[feature_name] = (
            gap_change / elapsed_hours
        ).where(
            elapsed_hours.gt(0)
            & elapsed_hours.notna()
            & gap_change.notna()
        )

    
    
    # 10C. Dominance stability over 12 consecutive hours.
    # The current pollutant is compared with the previous
    # 11 hourly dominant-pollutant labels.

    window_size = 12

    dominant = (
        df["Dominant_Pollutant"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    # Convert each label into a numeric indicator.
    # This allows pandas rolling operations to work safely.
    stability_parts = []

    for pollutant in POLLUTANTS:
        matches = dominant.eq(pollutant).astype("float64")

        # Missing labels must remain missing, not become zero.
        matches = matches.mask(dominant.isna())

        stability_parts.append(matches.rename(pollutant))

    match_matrix = pd.concat(stability_parts, axis=1)

    # For each hour, calculate the fraction of the last
    # 12 valid labels matching that hour's dominant pollutant.
    valid_window = (
        dominant.notna()
        .rolling(window_size, min_periods=window_size)
        .sum()
        .eq(window_size)
    )

    stability_values = match_matrix.rolling(
        window=window_size,
        min_periods=window_size,
    ).mean()

    # Select the score corresponding to the current
    # dominant pollutant.
    current_match = pd.Series(
        float("nan"),
        index=df.index,
        dtype="float64",
    )

    for pollutant in POLLUTANTS:
        current_mask = dominant.eq(pollutant)

        current_match.loc[current_mask] = (
            stability_values.loc[current_mask, pollutant]
        )

    df["Dominance_Stability_12h"] = (
        current_match.where(valid_window)
    )


    # --------------------------------------------------
    # 10. HANDLE UNAVAILABLE AQI ROWS
    # --------------------------------------------------

    aqi_boundary_columns = [
        "AQI_Boundary_Distance",
        "AQI_Signed_Boundary_Distance",
        "Nearest_AQI_Boundary",
    ]

    df.loc[
        df["AQI"].isna(),
        aqi_boundary_columns
    ] = None

    # --------------------------------------------------
    # 11. SAVE OUTPUT
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # 12. SUMMARY REPORT
    # --------------------------------------------------

    print("\nRegulatory feature extraction completed.")
    print("-----------------------------------------")

    print(f"Total rows: {len(df):,}")

    print(
        "Rows with available AQI:",
        df["AQI"].notna().sum()
    )

    print(
        "Rows with missing AQI:",
        df["AQI"].isna().sum()
    )

    print(
        "Rows with missing AQI boundary distances:",
        df["AQI_Boundary_Distance"].isna().sum()
    )

    print(
        "Rows with tied dominant pollutants:",
        df["Tied_Dominant_Pollutants"].notna().sum()
        if "Tied_Dominant_Pollutants" in df.columns
        else "Tie column not present"
    )

    print("\nCo-elevation counts above 50:")
    print(
        df["CoElevated_Count_Above_50"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCo-elevation counts above 100:")
    print(
        df["CoElevated_Count_Above_100"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValid sub-index count:")
    print(
        df["Valid_Subindex_Count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSelected feature statistics:")

    summary_columns = [
        "AQI_Boundary_Distance",
        "AQI_Signed_Boundary_Distance",
        "CoElevated_Count_Above_50",
        "CoElevated_Count_Above_100",
        "Valid_Subindex_Count",
        "Dominance_Gap",
    ]

    print(
        df[summary_columns]
        .describe()
        .to_string()
    )

    print(f"\nOutput saved to:\n{OUTPUT_FILE}")


# --------------------------------------------------
# 13. RUN SCRIPT
# --------------------------------------------------

if __name__ == "__main__":
    main()