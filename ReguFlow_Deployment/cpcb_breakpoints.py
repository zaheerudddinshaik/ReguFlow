
"""
AQI breakpoint configuration matching the uploaded
AQI-Calculator.xlsx workbook.

This configuration is for comparison and project testing.
Verify all breakpoint conventions against official CPCB
methodology before claiming full regulatory compliance.
"""

# Each interval:
# (concentration_low, concentration_high, aqi_low, aqi_high)

CPCB_BREAKPOINTS = {
    "PM2.5 (µg/m³)": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 380, 401, 500),
    ],

    "PM10 (µg/m³)": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 510, 401, 500),
    ],

    "NO2 (µg/m³)": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 520, 401, 500),
    ],

    "SO2 (µg/m³)": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2620, 401, 500),
    ],

    "CO (mg/m³)": [
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500),
    ],

    "Ozone (µg/m³)": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
}

# Upper limits represented in the workbook.
SEVERE_THRESHOLDS = {
    "PM2.5 (µg/m³)": 380,
    "PM10 (µg/m³)": 510,
    "NO2 (µg/m³)": 520,
    "SO2 (µg/m³)": 2620,
    "CO (mg/m³)": 50.0,
    "Ozone (µg/m³)": 1000,
}

AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

AVERAGING_HOURS = {
    "PM2.5 (µg/m³)": 24,
    "PM10 (µg/m³)": 24,
    "NO2 (µg/m³)": 24,
    "SO2 (µg/m³)": 24,
    "CO (mg/m³)": 8,
    "Ozone (µg/m³)": 8,
}

if __name__ == "__main__":
    print("AQI breakpoint configuration loaded.")
    print("Pollutants:", list(CPCB_BREAKPOINTS.keys()))
    print("Averaging periods:", AVERAGING_HOURS)
