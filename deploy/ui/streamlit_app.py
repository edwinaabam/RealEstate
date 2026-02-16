import streamlit as st
import pandas as pd
import requests
from PIL import Image
import json
import os




st.set_page_config(page_title="AlloyTower RealEstate", layout="wide")


# ==========================================
# API CONFIGURATION (Professional Setup)
# ==========================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# ==================================================
# 🟢 Sidebar Logo
# ==================================================

logo = Image.open("logo.png")
st.sidebar.image(logo, width="stretch")
st.sidebar.markdown("## AlloyTower Real Estate")

# ==================================================
# Top Banner (CENTERED)
# ==================================================

st.markdown("""
    <div style="
        background-color:#1e7f4f;
        padding:25px;
        border-radius:10px;
        text-align:center;
    ">
        <h1 style="color:white; margin-bottom:5px;">
            AlloyTower Real Estate
        </h1>
        <p style="color:white; margin-top:5px; font-size:18px;">
            Intelligent Property Valuation & Price Forecasting Platform
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================================================
# Top Navigation Tabs
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Market Insights Dashboard",
    "🏠 Property Valuation",
    "📈 Price Forecast"
])

# ==================================================
# 1️⃣ DASHBOARD PAGE
# ==================================================

with tab1:

    st.subheader("AlloyTower Analytics Dashboard")

    powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=c0f12726-823d-4cf8-b7fd-0a575bec286a&autoAuth=true&ctid=5f3c9581-2d3f-469e-acf3-b6065c0e681a&actionBarEnabled=true"

    st.components.v1.iframe(
        powerbi_url,
        height=700,
        scrolling=True
    )

# ==================================================
# 2️⃣ PROPERTY VALUATION PAGE
# ==================================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

with tab2:

    st.subheader("Property Valuation Engine")

    # --------------------------------------------------
    # Load FULL Feature Dataset
    # --------------------------------------------------

    x_tree = pd.read_csv(
    os.path.join(PROJECT_ROOT, "model", "x_tree.csv")
        )

    x_tree["PROPERTY_ID"] = x_tree["PROPERTY_ID"].astype(str)

    # --------------------------------------------------
    # Load Location Data
    # --------------------------------------------------

    #df_location = pd.read_csv(
     #   r"C:\Users\Edwina\Documents\Amdari_files\ds_projects\CentralizedDataPlatform\database\raw_data\DIM_PROPERTY_LOCATION.csv"
    #)

    df_location = pd.read_csv(
    os.path.join(PROJECT_ROOT, "database", "raw_data", "DIM_PROPERTY_LOCATION.csv")
        )

    df_location.columns = df_location.columns.str.strip().str.upper()
    df_location["PROPERTY_ID"] = df_location["PROPERTY_ID"].astype(str)

    # Merge
    df = x_tree.merge(df_location, on="PROPERTY_ID", how="left")

    # Load feature schema
    #with open(
    #    r"C:\Users\Edwina\Documents\Amdari_files\ds_projects\CentralizedDataPlatform\model\feature_schema.json"
    #) as f:
     #   feature_columns = json.load(f)

    with open(
    os.path.join(PROJECT_ROOT, "model", "feature_schema.json")
    ) as f:
        feature_columns = json.load(f)

    # =====================================================
    # LOCATION FILTERS
    # =====================================================

    st.markdown("### Location")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("State", sorted(df["STATE"].dropna().unique()))

    df_state = df[df["STATE"] == state]

    with col2:
        county = st.selectbox("County", sorted(df_state["COUNTY"].dropna().unique()))

    df_county = df_state[df_state["COUNTY"] == county]

    col3, col4 = st.columns(2)

    with col3:
        zipcode = st.selectbox(
            "Zip Code",
            sorted(df_county["ZIP_CODE"].dropna().unique())
        )

    df_zip = df_county[df_county["ZIP_CODE"] == zipcode]

    # =====================================================
    # PROPERTY TYPE
    # =====================================================

    st.markdown("### Property Type")

    property_type = st.selectbox(
        "Property Type",
        sorted(df_zip["PROPERTY_TYPE"].dropna().unique())
    )

    df_type = df_zip[df_zip["PROPERTY_TYPE"] == property_type]

    # =====================================================
    # PROPERTY FEATURES FILTERS
    # =====================================================

    st.markdown("### Property Features")

    col5, col6 = st.columns(2)

    with col5:
        bedrooms = st.selectbox(
            "Bedrooms",
            sorted(df_type["BEDROOMS"].unique())
        )

    df_bed = df_type[df_type["BEDROOMS"] == bedrooms]

    with col6:
        bathrooms = st.selectbox(
            "Bathrooms",
            sorted(df_bed["BATHROOMS"].unique())
        )

    df_bath = df_bed[df_bed["BATHROOMS"] == bathrooms]

    col7, col8 = st.columns(2)

    with col7:
        sqft = st.selectbox(
            "Square Footage",
            sorted(df_bath["SQUARE_FOOTAGE"].unique())
        )

    df_sq = df_bath[df_bath["SQUARE_FOOTAGE"] == sqft]

    with col8:
        lot_size = st.selectbox(
            "Lot Size",
            sorted(df_sq["LOT_SIZE"].unique())
        )

    df_final = df_sq[df_sq["LOT_SIZE"] == lot_size]

    st.divider()

    # =====================================================
    # VALUATION OUTPUT
    # =====================================================

    if df_final.empty:
        st.warning("No properties match selected criteria.")
    else:

        # ----------- Summary Cards -----------
        st.markdown("### Property Summary")

        colA, colB, colC, colD = st.columns(4)

        colA.metric("Bedrooms", bedrooms)
        colB.metric("Bathrooms", bathrooms)
        colC.metric("Sqft", sqft)
        colD.metric("Lot Size", lot_size)

        st.divider()

        if st.button("Estimate Property Value"):

            predictions = []

            for _, row in df_final.iterrows():

                valid_features = {
                    col: float(row[col]) if col in row else 0.0
                    for col in feature_columns
                }

                payload = {
                    "PROPERTY_TYPE": property_type,
                    "features": valid_features
                }

                response = requests.post(
                    f"{API_URL}/predict_valuation/1",
                    json=payload,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    predictions.append(result["predicted_property_value"])

            if predictions:

                avg_prediction = sum(predictions) / len(predictions)

                st.metric(
                    label=f"Estimated Value ({property_type} in {county}, {state})",
                    value=f"£{avg_prediction:,.0f}"
                )
            else:
                st.error("Prediction failed.")

# ==================================================
# 3️⃣ PRICE FORECAST PAGE
# ==================================================

with tab3:

    st.subheader("Market Price Forecast Engine")

    # --------------------------------------------------
    # Forecast Context (UI Only)
    # --------------------------------------------------

    df_location = pd.read_csv(
        r"C:\Users\Edwina\Documents\Amdari_files\ds_projects\CentralizedDataPlatform\database\raw_data\DIM_PROPERTY_LOCATION.csv"
    )

    st.markdown("### Forecast Context")

    col1, col2 = st.columns(2)

    with col1:
        forecast_state = st.selectbox(
            "State",
            sorted(df_location["STATE"].dropna().unique()),
            key="forecast_state"
        )

    with col2:
        forecast_property_type = st.selectbox(
            "Property Type",
            ["Single Family", "Condo", "Multi-Family"],
            key="forecast_property_type"
        )

    st.divider()

    # --------------------------------------------------
    # Forecast Settings
    # --------------------------------------------------

    st.markdown("### Forecast Settings")

    forecast_horizon = st.selectbox(
        "Forecast Horizon (Months)",
        options=[3, 6, 9, 12, 18, 24, 30, 36],
        index=3,
        key="forecast_horizon"
    )

    st.divider()

    # --------------------------------------------------
    # Run Forecast
    # --------------------------------------------------

    if st.button("Generate Market Forecast", key="forecast_button"):

        response = requests.get(
            f"{API_URL}/forecast_price?steps={forecast_horizon}",
            timeout=10
        )

        if response.status_code == 200:

            data = response.json()
            forecast_values = data["forecast_values"]

            projected_price = forecast_values[-1]
            avg_forecast_price = sum(forecast_values) / len(forecast_values)

            growth_absolute = forecast_values[-1] - forecast_values[0]
            growth_percent = (growth_absolute / forecast_values[0]) * 100

            # --------------------------------------------------
            # Summary Section
            # --------------------------------------------------

            st.markdown("### Forecast Summary")

            colA, colB, colC = st.columns(3)

            colA.metric(
                f"Projected Price ({forecast_horizon} Months)",
                f"${projected_price:,.0f}"
            )

            colB.metric(
                "Average Forecasted Price",
                f"${avg_forecast_price:,.0f}"
            )

            colC.metric(
                "Projected Growth",
                f"{growth_percent:.2f}%",
                f"${growth_absolute:,.0f}"
            )

            st.divider()

            # --------------------------------------------------
            # Insight Box
            # --------------------------------------------------

            if growth_percent > 0:
                trend_message = "Market is projected to grow."
            elif growth_percent < 0:
                trend_message = "Market is projected to decline."
            else:
                trend_message = "Market is projected to remain stable."

            st.info(
                f"""
                📈 **Market Insight**

                The {forecast_property_type} segment in {forecast_state}
                is projected to change by **{growth_percent:.2f}%**
                over the next **{forecast_horizon} months**.

                {trend_message}
                """
            )

        else:
            st.error("Forecast failed.")
