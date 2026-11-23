
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "HB_Colony_Vijayawada_subindices_candidate.csv"
OUTPUT_FILE = BASE_DIR / "HB_Colony_Vijayawada_aqi_candidate.csv"


# --------------------------------------------------
# 2. POLLUTANT CONFIGURATION
# --------------------------------------------------

# Keys are the pollutant names used in the output.
# Values are the corresponding sub-index column names.

POLLUTANTS = {
    "PM2.5": "PM2.5 (µg/m³)__subindex",
    "PM10": "PM10 (µg/m³)__subindex",
    "NO2": "NO2 (µg/m³)__subindex",
    "SO2": "SO2 (µg/m³)__subindex",
    "CO": "CO (mg/m³)__subindex",
    "Ozone": "Ozone (µg/m³)__subindex",
}


# --------------------------------------------------
# 3. AQI CATEGORY FUNCTION
# --------------------------------------------------

def get_category(aqi):
    """
    Assigns an AQI category based on the configured
    project thresholds.
    """

    if pd.isna(aqi):
        return None

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    elif aqi <= 500:
        return "Severe"

    return None


# --------------------------------------------------
# 4. OVERALL AQI CALCULATION
# --------------------------------------------------

def calculate_row_aqi(row):
    """
    Calculates the overall AQI and pollutant dominance
    for one timestamp.

    Project eligibility rule:
    - At least 3 valid pollutant sub-indices
    - At least one of PM2.5 or PM10 must be valid
    """

    valid_subindices = {}

    # Collect valid sub-indices
    for pollutant, column in POLLUTANTS.items():

        value = row[column]

        if pd.notna(value):
            valid_subindices[pollutant] = float(value)

    valid_count = len(valid_subindices)

    # PM requirement
    pm_available = (
        "PM2.5" in valid_subindices
        or "PM10" in valid_subindices
    )

    # Check eligibility
    if valid_count < 3 or not pm_available:

        return pd.Series({
            "AQI": None,
            "AQI_Category": None,
            "Dominant_Pollutant": None,
            "Dominant_Subindex": None,
            "Second_Dominant_Pollutant": None,
            "Second_Dominant_Subindex": None,
            "Dominance_Gap": None,
            "Tied_Dominant_Pollutants": None,
            "Valid_Subindex_Count": valid_count,
            "AQI_Eligible": False,
        })

    # Sort valid sub-indices in descending order.
    # Python's stable sorting makes tie ordering deterministic
    # according to the order in POLLUTANTS.
    sorted_subindices = sorted(
        valid_subindices.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Highest sub-index determines the overall AQI
    dominant_pollutant, dominant_value = sorted_subindices[0]

    # AQI sub-indices are expected to be rounded integers
    aqi = int(round(dominant_value))

    # Second-highest pollutant and sub-index
    if len(sorted_subindices) >= 2:

        second_pollutant, second_value = sorted_subindices[1]
        dominance_gap = dominant_value - second_value

    else:
        second_pollutant = None
        second_value = None
        dominance_gap = None

    # Identify all pollutants tied at the highest value
    tied_pollutants = [
        pollutant
        for pollutant, value in sorted_subindices
        if value == dominant_value
    ]

    tied_dominant_pollutants = (
        ", ".join(tied_pollutants)
        if len(tied_pollutants) > 1
        else None
    )

    # Determine AQI category
    aqi_category = get_category(aqi)

    return pd.Series({
        "AQI": aqi,
        "AQI_Category": aqi_category,
        "Dominant_Pollutant": dominant_pollutant,
        "Dominant_Subindex": dominant_value,
        "Second_Dominant_Pollutant": second_pollutant,
        "Second_Dominant_Subindex": second_value,
        "Dominance_Gap": dominance_gap,
        "Tied_Dominant_Pollutants": tied_dominant_pollutants,
        "Valid_Subindex_Count": valid_count,
        "AQI_Eligible": True,
    })


# --------------------------------------------------
# 5. MAIN PROCESSING FUNCTION
# --------------------------------------------------

def main():

    print("Starting overall AQI calculation...")

    # Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # Load sub-index candidate dataset
    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows: {len(df):,}")

    # Check required sub-index columns
    required_columns = list(POLLUTANTS.values())

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required sub-index columns "
            f"are missing: {missing_columns}"
        )

    # Convert sub-index values to numeric.
    # Invalid text values become NaN and are not used
    # in the AQI calculation.
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Reject invalid sub-index values
    for column in required_columns:

        invalid = (
            df[column].notna()
            & (
                (df[column] < 0)
                | (df[column] > 500)
            )
        )

        if invalid.any():
            raise ValueError(
                f"Invalid sub-index values found in {column}. "
                "Expected values between 0 and 500."
            )

    # Calculate overall AQI and regulatory fields
    aqi_results = df.apply(
        calculate_row_aqi,
        axis=1
    )

    # Add calculated fields to original dataset
    df = pd.concat(
        [df, aqi_results],
        axis=1
    )

    # Save candidate dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # 6. SUMMARY REPORT
    # --------------------------------------------------

    total_rows = len(df)
    eligible_rows = int(df["AQI_Eligible"].sum())
    ineligible_rows = total_rows - eligible_rows

    print("\nAQI calculation completed successfully.")
    print("-----------------------------------------")
    print(f"Total rows: {total_rows:,}")
    print(f"AQI eligible rows: {eligible_rows:,}")
    print(f"AQI ineligible rows: {ineligible_rows:,}")

    print("\nAQI category counts:")

    category_counts = (
        df["AQI_Category"]
        .value_counts(dropna=False)
    )

    print(category_counts.to_string())

    print("\nDominant pollutant counts:")

    dominant_counts = (
        df["Dominant_Pollutant"]
        .value_counts(dropna=False)
    )

    print(dominant_counts.to_string())

    tied_count = int(
        df["Tied_Dominant_Pollutants"].notna().sum()
    )

    print(f"\nRows with tied dominant pollutants: {tied_count:,}")

    print(f"\nOutput saved to:\n{OUTPUT_FILE}")


# --------------------------------------------------
# 7. RUN SCRIPT
# --------------------------------------------------

if __name__ == "__main__":
    main()