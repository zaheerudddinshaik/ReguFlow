
import warnings
import pandas as pd
import streamlit as st

warnings.filterwarnings(
    "ignore",
    category=pd.errors.PerformanceWarning,
)

from reguflow_openmeteo_adapter import fetch_current_inputs
from reguflow_live_regulatory import build_live_aqi_features
from regulatory_state_live_features import build_regulatory_state_features
from reguflow_current_forecast import forecast_current

st.set_page_config(
    page_title='Forecasting Hidden Multi-Pollutant Regulatory States Beneath Future AQI',
    page_icon="??",
    layout="wide",
)

st.title('Forecasting Hidden Multi-Pollutant Regulatory States Beneath Future AQI')
st.subheader("ReguFlow Current-Data Air Quality Forecasting")

st.info(
    "Experimental forecasting using Open-Meteo model-estimated "
    "air-quality and meteorological inputs. "
    "Not live CPCB station measurements or an officially "
    "validated regulatory AQI forecast."
)

POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Ozone",
]

@st.cache_data(ttl=1800, show_spinner=False)
def get_forecasts():
    df = fetch_current_inputs()
    df = build_live_aqi_features(df)
    df = build_regulatory_state_features(df)
    predictions = forecast_current(df)
    return predictions, df

with st.sidebar:
    st.header("Forecast Controls")

    model = st.selectbox(
        "Model",
        ["Original ReguFlow", "Coupled ReguFlow"],
    )

    pollutant = st.selectbox(
        "Pollutant",
        POLLUTANTS,
    )

    if st.button("Refresh current data", use_container_width=True):
        get_forecasts.clear()
        st.rerun()

try:
    with st.spinner("Fetching data and generating forecasts..."):
        predictions, inputs = get_forecasts()

except Exception as exc:
    st.error("Unable to generate forecasts.")
    st.exception(exc)
    st.stop()

issue_time = predictions["Forecast_Issue_Time"].iloc[0]

st.caption(
    f"Location: HB Colony, Vijayawada | "
    f"Latest input: {issue_time} IST | "
    f"Source: Open-Meteo"
)

selected = (
    predictions[predictions["Model"] == model]
    .sort_values("Horizon_hours")
    .copy()
)

col1, col2, col3, col4 = st.columns(4)

for column, horizon in zip(
    [col1, col2, col3, col4],
    [1, 6, 12, 24],
):
    value = selected.loc[
        selected["Horizon_hours"] == horizon,
        pollutant,
    ].iloc[0]

    unit = "mg/m?" if pollutant == "CO" else "µg/m³"

    column.metric(
        f"{horizon}-hour forecast",
        f"{value:.2f} {unit}",
    )

st.divider()

st.subheader(f"{model}: {pollutant} Forecast")

chart_data = selected[
    ["Horizon_hours", pollutant]
].set_index("Horizon_hours")

st.line_chart(chart_data)

st.caption(
    "Chart points represent separate direct forecast horizons, "
    "not a continuous hourly prediction trajectory."
)

st.subheader("All Predicted Pollutants")

display_columns = [
    "Horizon_hours",
    "Target_Time",
    *POLLUTANTS,
]

st.dataframe(
    selected[display_columns].round(3),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Original vs Coupled ReguFlow")

comparison = predictions.pivot(
    index="Horizon_hours",
    columns="Model",
    values=pollutant,
)

st.dataframe(
    comparison.round(3),
    use_container_width=True,
)


st.divider()

st.divider()
st.subheader("Future Regulatory Forecasts")

st.caption(
    "Scenario-based future AQI and regulatory-state estimates "
    "at 1, 6, 12, and 24 hours. Intermediate hourly pollutant "
    "values are interpolated between direct ReguFlow forecasts."
)

from reguflow_future_regulatory import (
    forecast_future_regulatory,
)

@st.cache_data(ttl=1800, show_spinner=False)
def get_future_regulatory(_history, _predictions):
    return forecast_future_regulatory(
        _history,
        _predictions,
    )

with st.spinner("Calculating future regulatory states..."):
    future = forecast_future_regulatory(
        inputs,
        predictions,
    )

future_selected = (
    future[
        (future["Model"] == model)
        & future["Horizon_hours"].isin([1, 6, 12, 24])
    ]
    .sort_values("Horizon_hours")
    .copy()
)

def aqi_category(value):
    if pd.isna(value):
        return "Unavailable"
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"

future_selected["AQI_Category"] = (
    future_selected["AQI"].apply(aqi_category)
)

st.markdown("#### Predicted AQI and Category")

cols = st.columns(4)

for col, (_, row) in zip(
    cols,
    future_selected.iterrows(),
):
    with col:
        st.metric(
            f"+{int(row['Horizon_hours'])}h AQI",
            f"{row['AQI']:.0f}",
        )
        st.caption(
            aqi_category(row["AQI"])
        )

st.markdown("#### Future AQI Trend")

aqi_chart = (
    future[future["Model"] == model]
    .set_index("Timestamp")[["AQI"]]
)

st.line_chart(aqi_chart)

st.caption(
    "The hourly AQI trajectory is reconstructed from "
    "interpolated concentrations, not 24 independent "
    "hourly ReguFlow predictions."
)

st.markdown("#### Future Dominant Pollutants")

st.dataframe(
    future_selected[
        [
            "Horizon_hours",
            "Timestamp",
            "AQI",
            "AQI_Category",
            "Dominant_Pollutant",
            "Dominance_Gap",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.markdown("#### Future Regulatory Dynamics")

dynamic_features = [
    "AQI_Boundary_Distance",
    "AQI_Boundary_Velocity",
    "Dominance_Stability_12h",
    "ReguFlow_Top2_Dominance_Gap",
    "ReguFlow_Competition_Flux",
]

selected_dynamic = st.selectbox(
    "Regulatory dynamic feature",
    dynamic_features,
    key="future_dynamic_feature",
)

st.line_chart(
    future[future["Model"] == model]
    .set_index("Timestamp")[
        [selected_dynamic]
    ]
)

st.markdown("#### All Future Regulatory Features")

from reguflow_inference_engine import load_model

_, regulatory_checkpoint = load_model(
    "Original ReguFlow",
    1,
)

future_regulatory_columns = [
    name
    for name in regulatory_checkpoint["regulatory_features"]
    if not name.endswith("__is_missing")
    and name in future_selected.columns
]

st.dataframe(
    future_selected[
        [
            "Horizon_hours",
            "Timestamp",
            *future_regulatory_columns,
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download future regulatory predictions (CSV)",
    data=future_selected.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="Forecasting Hidden Multi-Pollutant Regulatory States Beneath Future AQI_future_regulatory.csv",
    mime="text/csv",
)


st.subheader("Regulatory State Analysis")

st.caption(
    "Historical regulatory features calculated from "
    "Open-Meteo model-estimated input data. "
    "These are not future regulatory-state forecasts."
)

latest = inputs.iloc[-1]

def display_value(name, digits=2):
    value = latest.get(name, float("nan"))
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Latest estimated AQI",
    display_value("AQI", 0),
)

r2.metric(
    "Dominant pollutant",
    str(latest.get("Dominant_Pollutant", "N/A")),
)

r3.metric(
    "Dominance gap",
    display_value("Dominance_Gap"),
)

r4.metric(
    "12h dominance stability",
    display_value("Dominance_Stability_12h"),
)

tabs = st.tabs([
    "Pollutant Subindices",
    "Boundary Geometry",
    "Velocity and Flux",
    "Dominance and Competition",
    "All Regulatory Features",
])

with tabs[0]:

    st.markdown("#### Pollutant Subindices")

    columns = [
        c for c in inputs.columns
        if c.endswith("__subindex")
    ]

    if columns:
        st.line_chart(
            inputs.set_index("Timestamp")[
                columns
            ].tail(72)
        )

        st.dataframe(
            inputs[
                ["Timestamp", *columns]
            ].tail(24),
            use_container_width=True,
            hide_index=True,
        )

with tabs[1]:

    st.markdown("#### Regulatory Boundary Geometry")

    columns = [
        c for c in inputs.columns
        if (
            "Boundary_Distance" in c
            or c == "Nearest_AQI_Boundary"
        )
    ]

    if columns:
        selected_feature = st.selectbox(
            "Boundary feature",
            columns,
            key="boundary_feature",
        )

        st.line_chart(
            inputs.set_index("Timestamp")[
                [selected_feature]
            ].tail(72)
        )

with tabs[2]:

    st.markdown("#### ReguFlow Velocity and Flux")

    columns = [
        c for c in inputs.columns
        if (
            "Boundary_Velocity" in c
            or "Boundary_Flux" in c
            or "ReguFlow_Signed_Distance" in c
            or c == "ReguFlow_Competition_Flux"
        )
    ]

    if columns:
        selected_feature = st.selectbox(
            "Dynamic regulatory feature",
            columns,
            key="dynamic_feature",
        )

        st.line_chart(
            inputs.set_index("Timestamp")[
                [selected_feature]
            ].tail(72)
        )

with tabs[3]:

    st.markdown("#### Dominance and Competition")

    columns = [
        "Dominance_Gap",
        "Dominance_Stability_12h",
        "ReguFlow_Top2_Dominance_Gap",
        "ReguFlow_Competition_Flux",
        "CoElevated_Count_Above_50",
        "CoElevated_Count_Above_100",
    ]

    columns = [
        c for c in columns
        if c in inputs.columns
    ]

    if columns:
        selected_feature = st.selectbox(
            "Dominance feature",
            columns,
            key="dominance_feature",
        )

        st.line_chart(
            inputs.set_index("Timestamp")[
                [selected_feature]
            ].tail(72)
        )

    if "Dominant_Pollutant" in inputs.columns:
        st.dataframe(
            inputs[
                ["Timestamp", "Dominant_Pollutant"]
            ].tail(24),
            use_container_width=True,
            hide_index=True,
        )

with tabs[4]:

    st.markdown("#### Complete Regulatory Feature Table")

    from reguflow_inference_engine import load_model

    _, checkpoint = load_model(
        "Original ReguFlow",
        1,
    )

    regulatory_names = [
        name
        for name in checkpoint["regulatory_features"]
        if not name.endswith("__is_missing")
    ]

    available = [
        name
        for name in regulatory_names
        if name in inputs.columns
    ]

    missing = [
        name
        for name in regulatory_names
        if name not in inputs.columns
    ]

    st.write(
        f"Available numerical regulatory features: "
        f"{len(available)} / {len(regulatory_names)}"
    )

    if missing:
        st.warning(
            "Unavailable features: " + ", ".join(missing)
        )

    st.dataframe(
        inputs[
            ["Timestamp", *available]
        ].tail(48),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download regulatory features (CSV)",
        data=inputs[
            ["Timestamp", *available]
        ].to_csv(index=False).encode("utf-8-sig"),
        file_name="Forecasting Hidden Multi-Pollutant Regulatory States Beneath Future AQI_regulatory_features.csv",
        mime="text/csv",
    )


st.subheader("Forecast Data Export")

csv = predictions.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "Download all predictions (CSV)",
    data=csv,
    file_name="Forecasting Hidden Multi-Pollutant Regulatory States Beneath Future AQI_current_forecasts.csv",
    mime="text/csv",
)

with st.expander("Data provenance and limitations"):
    st.write(
        "Air-quality inputs are Open-Meteo model estimates. "
        "Meteorological inputs come from Open-Meteo. "
        "The models were trained on historical station data, "
        "so cross-source predictive accuracy has not been validated."
    )
    st.write(
        "AQI-derived features use the project's existing "
        "candidate regulatory calculations. "
        "Forecast concentrations should not be interpreted "
        "as certified CPCB AQI values."
    )
    st.write(
        "Input freshness and unit mappings should be checked "
        "before operational use."
    )




# FINAL_REGULATORY_FORECAST_FEATURES_V1

st.divider()
st.header("Future AQI Transitions and Regulatory Risk")

st.caption(
    "Experimental regulatory scenarios reconstructed from "
    "direct 1h, 6h, 12h and 24h pollutant forecasts. "
    "Intermediate hours are interpolated."
)

import numpy as np

FINAL_HORIZONS = [1, 6, 12, 24]
FINAL_BOUNDARIES = [50, 100, 200, 300, 400, 500]


def final_aqi_category(value):
    if pd.isna(value):
        return "Unavailable"
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def final_category_rank(value):
    categories = [
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "Very Poor",
        "Severe",
    ]
    return categories.index(value)


def final_transition(previous, current):
    if pd.isna(previous) or pd.isna(current):
        return "Unavailable"

    previous_category = final_aqi_category(previous)
    current_category = final_aqi_category(current)

    difference = (
        final_category_rank(current_category)
        - final_category_rank(previous_category)
    )

    if difference > 0:
        return "Worsening"
    if difference < 0:
        return "Improving"
    return "Same category"


def final_boundary_proximity(value):
    if pd.isna(value):
        return np.nan

    return min(
        abs(float(value) - boundary)
        for boundary in FINAL_BOUNDARIES
    )


def final_boundary_status(value):
    distance = final_boundary_proximity(value)

    if pd.isna(distance):
        return "Unavailable"
    if distance <= 5:
        return "Near boundary"
    if distance <= 15:
        return "Approaching boundary"
    return "Away from boundary"


required_columns = [
    "Model",
    "Horizon_hours",
    "Timestamp",
    "AQI",
    "Dominant_Pollutant",
]

missing_columns = [
    c for c in required_columns
    if c not in future.columns
]

if missing_columns:
    st.error(
        "Missing future regulatory columns: "
        + str(missing_columns)
    )
    st.stop()

final_data = future[
    future["Horizon_hours"].isin(FINAL_HORIZONS)
].copy()

final_data["Horizon_hours"] = (
    final_data["Horizon_hours"].astype(int)
)

final_data["AQI_Category"] = (
    final_data["AQI"].apply(final_aqi_category)
)

final_data["Boundary_Proximity"] = (
    final_data["AQI"].apply(final_boundary_proximity)
)

final_data["Boundary_Status"] = (
    final_data["AQI"].apply(final_boundary_status)
)

final_data = final_data.sort_values(
    ["Model", "Horizon_hours"]
).reset_index(drop=True)

final_data["Previous_AQI"] = (
    final_data.groupby("Model")["AQI"].shift(1)
)

final_data["Previous_Dominant"] = (
    final_data.groupby("Model")[
        "Dominant_Pollutant"
    ].shift(1)
)

final_data["Category_Transition"] = [
    "First forecast"
    if pd.isna(previous)
    else final_transition(previous, current)
    for previous, current in zip(
        final_data["Previous_AQI"],
        final_data["AQI"],
    )
]

final_data["Dominant_Pollutant_Change"] = [
    "First forecast"
    if pd.isna(previous)
    else (
        "Changed"
        if previous != current
        else "Unchanged"
    )
    for previous, current in zip(
        final_data["Previous_Dominant"],
        final_data["Dominant_Pollutant"],
    )
]

# -----------------------------------------
# 1. Category transitions
# -----------------------------------------

st.subheader("1. Future AQI Category Transitions")

selected_final = final_data[
    final_data["Model"] == model
].copy()

st.dataframe(
    selected_final[
        [
            "Horizon_hours",
            "Timestamp",
            "AQI",
            "AQI_Category",
            "Category_Transition",
            "Dominant_Pollutant",
            "Dominant_Pollutant_Change",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Category transitions compare consecutive displayed "
    "forecast horizons, not every intermediate hour."
)

# -----------------------------------------
# 2. Regulatory boundary risk
# -----------------------------------------

st.subheader("2. Future Regulatory Boundary Risk")

risk_columns = [
    "Horizon_hours",
    "AQI",
    "AQI_Category",
    "Boundary_Proximity",
    "Boundary_Status",
]

for optional_column in [
    "AQI_Boundary_Distance",
    "AQI_Boundary_Velocity",
    "Dominance_Gap",
    "Dominance_Stability_12h",
    "ReguFlow_Competition_Flux",
]:
    if optional_column in selected_final.columns:
        risk_columns.append(optional_column)

st.dataframe(
    selected_final[risk_columns],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Boundary proximity is the absolute distance "
    "from the nearest configured AQI threshold. "
    "Near boundary means within 5 AQI points; "
    "approaching boundary means within 15. "
    "These are descriptive indicators, not "
    "calibrated probabilities of category change."
)

# -----------------------------------------
# 3. Original vs Coupled comparison
# -----------------------------------------

st.subheader("3. Original vs Coupled ReguFlow")

comparison = final_data.pivot(
    index="Horizon_hours",
    columns="Model",
    values="AQI",
)

st.line_chart(comparison)

st.markdown("#### Forecast Comparison")

comparison_table = final_data[
    [
        "Model",
        "Horizon_hours",
        "AQI",
        "AQI_Category",
        "Dominant_Pollutant",
        "Boundary_Proximity",
    ]
]

st.dataframe(
    comparison_table,
    use_container_width=True,
    hide_index=True,
)

if {
    "Original ReguFlow",
    "Coupled ReguFlow",
}.issubset(comparison.columns):

    disagreement = pd.DataFrame({
        "Horizon_hours": comparison.index,
        "Original_AQI": comparison[
            "Original ReguFlow"
        ].values,
        "Coupled_AQI": comparison[
            "Coupled ReguFlow"
        ].values,
    })

    disagreement["Absolute_AQI_Difference"] = (
        disagreement["Original_AQI"]
        - disagreement["Coupled_AQI"]
    ).abs()

    st.markdown("#### Model Disagreement")

    st.dataframe(
        disagreement,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Model disagreement measures differences "
        "between forecasts. It is not a calibrated "
        "uncertainty interval."
    )

# -----------------------------------------
# 4. Automated integrity checks
# -----------------------------------------

st.subheader("4. Forecast Integrity Checks")

checks = {}

checks["Both models available"] = {
    "Original ReguFlow",
    "Coupled ReguFlow",
}.issubset(set(final_data["Model"]))

checks["Four horizons per model"] = all(
    set(group["Horizon_hours"]) == set(FINAL_HORIZONS)
    for _, group in final_data.groupby("Model")
)

checks["Eight horizon predictions"] = (
    len(final_data) == 8
)

checks["No duplicate model/horizon pairs"] = (
    not final_data.duplicated(
        ["Model", "Horizon_hours"]
    ).any()
)

checks["All forecast AQI values present"] = (
    final_data["AQI"].notna().all()
)

checks["All forecast AQI values finite"] = (
    np.isfinite(
        pd.to_numeric(
            final_data["AQI"],
            errors="coerce",
        )
    ).all()
)

checks["All forecast AQI values nonnegative"] = (
    final_data["AQI"].ge(0).all()
)

checks["All dominant pollutants present"] = (
    final_data["Dominant_Pollutant"]
    .notna()
    .all()
)

checks["24 scenario hours per model"] = all(
    group["Timestamp"].nunique() == 24
    for _, group in future.groupby("Model")
)

checks["No duplicate scenario timestamps"] = (
    not future.duplicated(
        ["Model", "Timestamp"]
    ).any()
)

check_table = pd.DataFrame(
    [
        {
            "Check": name,
            "Status": "PASS" if passed else "FAIL",
        }
        for name, passed in checks.items()
    ]
)

st.dataframe(
    check_table,
    use_container_width=True,
    hide_index=True,
)

if all(checks.values()):
    st.success(
        "All structural forecast integrity checks passed."
    )
else:
    st.error(
        "Some forecast integrity checks failed. "
        "Review the table before using these results."
    )

st.caption(
    "Passing these checks confirms structural consistency, "
    "not regulatory compliance or predictive accuracy."
)

# -----------------------------------------
# 5. Export
# -----------------------------------------

st.subheader("5. Export Final Forecast Results")

st.download_button(
    "Download AQI transitions and regulatory risk",
    data=final_data.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="future_regulatory_risk_results.csv",
    mime="text/csv",
    key="final_regulatory_export",
)
