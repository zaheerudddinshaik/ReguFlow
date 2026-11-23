# ============================================================
# AQISTRATA + REGUFLOW
# CPCB Rolling-Average Generation
# ============================================================

from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = Path(
    r"C:\MINIProject\missing_value_handling"
    r"\HB_Colony_Vijayawada_missing_handling_candidate.csv"
)

OUTPUT_FILE = Path(
    r"C:\MINIProject\cpcb_preprocessing"
    r"\HB_Colony_Vijayawada_rolling_averages.csv"
)


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

TIMESTAMP_COL = "Timestamp"

POLLUTANT_WINDOWS = {
    "PM2.5 (µg/m³)": 24,
    "PM10 (µg/m³)": 24,
    "NO2 (µg/m³)": 24,
    "SO2 (µg/m³)": 24,
    "CO (mg/m³)": 8,
    "Ozone (µg/m³)": 8,
}

# Project-configured minimum valid observations.
# These thresholds are not independently certified as
# equivalent to official CPCB requirements.
MIN_VALID = {
    24: 16,
    8: 6,
}


# ------------------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# ------------------------------------------------------------

def load_data():
    """Load input data and validate timestamps and columns."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n"
            "Please check the input folder and filename."
        )

    df = pd.read_csv(INPUT_FILE)

    # Remove accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    # Check duplicate column names.
    if df.columns.duplicated().any():
        duplicates = df.columns[
            df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names found: {duplicates}"
        )

    # Check timestamp column.
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Timestamp column '{TIMESTAMP_COL}' not found."
        )

    # Convert timestamps.
    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL],
        errors="coerce"
    )

    if df[TIMESTAMP_COL].isna().any():
        count = int(df[TIMESTAMP_COL].isna().sum())

        raise ValueError(
            f"{count} invalid timestamps found."
        )

    # Sort chronologically.
    df = df.sort_values(
        TIMESTAMP_COL
    ).reset_index(drop=True)

    # Check duplicate timestamps.
    if df[TIMESTAMP_COL].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found. "
            "Resolve them before calculating rolling averages."
        )

    # Check continuous hourly timeline.
    differences = df[TIMESTAMP_COL].diff().dropna()

    invalid_intervals = differences[
        differences != pd.Timedelta(hours=1)
    ]

    if not invalid_intervals.empty:
        raise ValueError(
            "Input data is not a continuous hourly timeline. "
            "Missing or irregular hourly intervals were found. "
            "Resolve them before calculating rolling averages."
        )

    # Validate required pollutant and missingness columns.
    required_columns = []

    for pollutant in POLLUTANT_WINDOWS:
        required_columns.append(pollutant)
        required_columns.append(
            f"{pollutant}__was_missing"
        )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    print("Input data loaded and validated.")
    print(f"Total rows: {len(df):,}")
    print(
        "Date range:",
        df[TIMESTAMP_COL].min(),
        "to",
        df[TIMESTAMP_COL].max()
    )

    return df


# ------------------------------------------------------------
# 4. VALIDATE MISSINGNESS FLAGS
# ------------------------------------------------------------

def validate_missing_flags(df):
    """
    Ensure missingness flags contain only 0 or 1
    and have no null values.
    """

    for pollutant in POLLUTANT_WINDOWS:
        flag_col = f"{pollutant}__was_missing"

        flags = pd.to_numeric(
            df[flag_col],
            errors="coerce"
        )

        if flags.isna().any():
            raise ValueError(
                f"Null or invalid values found in {flag_col}."
            )

        if not flags.isin([0, 1]).all():
            raise ValueError(
                f"Unexpected values found in {flag_col}. "
                "Only 0 and 1 are allowed."
            )

        df[flag_col] = flags.astype("int8")

    print("Missingness flags validated.")

    return df


# ------------------------------------------------------------
# 5. GENERATE ROLLING AVERAGES
# ------------------------------------------------------------

def generate_rolling_averages(df):
    """
    Calculate trailing rolling averages using only
    originally observed pollutant measurements.

    Each rolling window includes the current timestamp
    and preceding hourly rows. Imputed values marked
    by a missingness flag of 1 are excluded.
    """

    df = df.copy()

    # Use timestamp as rolling index.
    df = df.set_index(TIMESTAMP_COL)

    for pollutant, window in POLLUTANT_WINDOWS.items():

        print(
            f"Processing {pollutant} "
            f"({window}-hour window)..."
        )

        # Convert pollutant measurements to numeric.
        values = pd.to_numeric(
            df[pollutant],
            errors="coerce"
        )

        # Read validated missingness flag.
        missing_flag = f"{pollutant}__was_missing"

        originally_missing = df[
            missing_flag
        ].eq(1)

        # Exclude originally missing or imputed readings.
        values = values.mask(originally_missing)

        # Count valid observations in the trailing window.
        valid_count = values.rolling(
            window=window,
            min_periods=1
        ).count()

        # Calculate rolling average only when the
        # configured minimum number of valid values exists.
        rolling_mean = values.rolling(
            window=window,
            min_periods=MIN_VALID[window]
        ).mean()

        # Store count and average.
        df[
            f"{pollutant}__{window}h_valid_count"
        ] = valid_count

        df[
            f"{pollutant}__{window}h_avg"
        ] = rolling_mean

    # Restore timestamp as a regular column.
    df = df.reset_index()

    return df


# ------------------------------------------------------------
# 6. SAVE RESULTS AND DISPLAY SUMMARY
# ------------------------------------------------------------

def save_results(df):
    """Save rolling averages and print summary."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nRolling averages generated successfully.")
    print(f"Total rows: {len(df):,}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nRolling Average Summary")
    print("-" * 65)

    for pollutant, window in POLLUTANT_WINDOWS.items():

        avg_col = f"{pollutant}__{window}h_avg"
        count_col = (
            f"{pollutant}__{window}h_valid_count"
        )

        valid_averages = int(
            df[avg_col].notna().sum()
        )

        missing_averages = int(
            df[avg_col].isna().sum()
        )

        print(f"\nPollutant: {pollutant}")
        print(f"Window: {window} hours")
        print(
            f"Minimum valid observations: "
            f"{MIN_VALID[window]}"
        )
        print(
            f"Valid averages: {valid_averages:,}"
        )
        print(
            f"Missing averages: {missing_averages:,}"
        )

        print(
            f"Maximum valid count: "
            f"{df[count_col].max()}"
        )

    print("\nProcessing completed.")


# ------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------

def main():

    print("=" * 65)
    print("AQISTRATA + REGUFLOW")
    print("CPCB ROLLING-AVERAGE GENERATION")
    print("=" * 65)

    # Step 1: Load and validate source data.
    df = load_data()

    # Step 2: Validate missingness flags.
    df = validate_missing_flags(df)

    # Step 3: Generate rolling averages.
    df = generate_rolling_averages(df)

    # Step 4: Save results and display summary.
    save_results(df)

    print("\n" + "=" * 65)
    print("ROLLING-AVERAGE PIPELINE FINISHED")
    print("=" * 65)

    print("\nImportant:")
    print("- Rolling windows are trailing and include current time.")
    print("- Originally missing or imputed values are excluded.")
    print("- Timestamp continuity is checked.")
    print("- Minimum valid counts use project-configured thresholds.")
    print("- Official CPCB compliance has not been independently certified.")


if __name__ == "__main__":
    main()