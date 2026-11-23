
from cpcb_breakpoints import CPCB_BREAKPOINTS, SEVERE_THRESHOLDS



import math

def calculate_subindex(pollutant, concentration):
    """
    Calculate a pollutant sub-index using the configured
    workbook breakpoint intervals.

    Uses the greatest lower concentration bound not exceeding
    the input, matching the uploaded Excel approximate lookup.
    Rounds positive values to the nearest integer.
    """

    if pollutant not in CPCB_BREAKPOINTS:
        raise ValueError(f"Unknown pollutant: {pollutant}")

    if concentration is None:
        return None

    try:
        concentration = float(concentration)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(concentration):
        return None

    if not 0 <= concentration <= SEVERE_THRESHOLDS[pollutant]:
        return None

    selected_interval = None

    for blo, bhi, ilo, ihi in CPCB_BREAKPOINTS[pollutant]:
        if blo <= concentration:
            selected_interval = (blo, bhi, ilo, ihi)
        else:
            break

    if selected_interval is None:
        return None

    blo, bhi, ilo, ihi = selected_interval

    if bhi <= blo:
        raise ValueError(
            f"Invalid breakpoint interval for {pollutant}: {blo}-{bhi}"
        )

    index = (
        ((ihi - ilo) / (bhi - blo))
        * (concentration - blo)
        + ilo
    )

    # Excel ROUND for non-negative values: round half up.
    return math.floor(index + 0.5)



if __name__ == "__main__":
    tests = [
        ("PM2.5 (µg/m³)", 45),
        ("PM10 (µg/m³)", 75),
        ("NO2 (µg/m³)", 60),
        ("SO2 (µg/m³)", 60),
        ("CO (mg/m³)", 1.5),
        ("Ozone (µg/m³)", 75),

        # Boundary checks for PM2.5
        ("PM2.5 (µg/m³)", 30),
        ("PM2.5 (µg/m³)", 30.5),
        ("PM2.5 (µg/m³)", 60),
        ("PM2.5 (µg/m³)", 60.5),
        ("PM2.5 (µg/m³)", 90),
        ("PM2.5 (µg/m³)", 90.5),

        # Invalid and unsupported values
        ("PM2.5 (µg/m³)", -5),
        ("PM2.5 (µg/m³)", None),
        ("PM2.5 (µg/m³)", 250),

        # Severe-range threshold checks
        ("PM2.5 (µg/m³)", 251),
        ("PM10 (µg/m³)", 431),
        ("NO2 (µg/m³)", 401),
        ("SO2 (µg/m³)", 1601),
        ("CO (mg/m³)", 35),
        ("Ozone (µg/m³)", 749),

        # CPCB reference sanity checks
        ("PM2.5 (µg/m³)", 31),  # Expected sub-index: 51
        ("PM2.5 (µg/m³)", 45),  # Expected sub-index: 75
        ("PM2.5 (µg/m³)", 60),  # Expected sub-index: 100
        ("PM10 (µg/m³)", 85),   # Expected sub-index: 85
    ]

    for pollutant, concentration in tests:
        result = calculate_subindex(pollutant, concentration)
        print(pollutant, concentration, "=>", result)
