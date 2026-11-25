
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from predict import predict_demand
from surplus import estimate_surplus


st.set_page_config(
    page_title="Smart Food Waste Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# DARK DASHBOARD DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #253047;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label, span {
    color: #CBD5E1;
}

[data-testid="stMetric"] {
    background-color: #151F32;
    border: 1px solid #26364F;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
}

div.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
    border: none;
}

[data-testid="stExpander"] {
    background-color: #151F32;
    border: 1px solid #26364F;
    border-radius: 10px;
}

hr {
    border-color: #26364F;
}

[data-testid="stCaptionContainer"] {
    color: #94A3B8;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    train = pd.read_csv(
        BASE_DIR / "data/raw/train.csv"
    )

    meals = pd.read_csv(
        BASE_DIR / "data/raw/meal_info.csv"
    )

    centers = pd.read_csv(
        BASE_DIR / "data/raw/fulfilment_center_info.csv"
    )

    return train, meals, centers


train, meals, centers = load_data()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    "<h1>🍃 Smart Food Waste Intelligence</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="color:#94A3B8; font-size:16px;">
    Machine learning-powered demand forecasting and
    inventory surplus intelligence.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Demand Analytics",
        "Model Performance",
        "Prediction Studio"
    ]
)

st.sidebar.divider()

st.sidebar.caption("SMART FOOD WASTE INTELLIGENCE")
st.sidebar.caption("Machine Learning Project")


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.subheader("Project Overview")

    total_orders = train["num_orders"].sum()
    total_records = len(train)
    total_centers = train["center_id"].nunique()
    total_meals = train["meal_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Historical Orders",
        f"{total_orders:,.0f}"
    )

    c2.metric(
        "Data Records",
        f"{total_records:,}"
    )

    c3.metric(
        "Fulfilment Centres",
        total_centers
    )

    c4.metric(
        "Unique Meals",
        total_meals
    )

    st.divider()

    st.subheader("Historical Demand Trend")

    weekly = (
        train.groupby("week")["num_orders"]
        .sum()
        .reset_index()
        .sort_values("week")
    )

    st.line_chart(
        weekly,
        x="week",
        y="num_orders",
        x_label="Week",
        y_label="Total Orders",
        color="#38BDF8"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Top Meal Categories")

        category_data = train.merge(
            meals,
            on="meal_id",
            how="left"
        )

        category_orders = (
            category_data.groupby("category")["num_orders"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )

        st.bar_chart(
            category_orders,
            horizontal=True,
            color="#2DD4BF"
        )

    with right:

        st.subheader("Top Fulfilment Centres")

        center_orders = (
            train.groupby("center_id")["num_orders"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )

        st.bar_chart(
            center_orders,
            horizontal=True,
            color="#818CF8"
        )

    st.caption(
        "These charts describe historical orders, not measured food waste."
    )


# --------------------------------------------------
# DEMAND ANALYTICS
# --------------------------------------------------

elif page == "Demand Analytics":

    st.subheader("Demand Analytics")

    st.write(
        "Explore historical food demand across weeks, "
        "meal categories and fulfilment centres."
    )

    min_week = int(train["week"].min())
    max_week = int(train["week"].max())

    selected_range = st.slider(
        "Select Week Range",
        min_value=min_week,
        max_value=max_week,
        value=(min_week, max_week)
    )

    filtered = train[
        train["week"].between(
            selected_range[0],
            selected_range[1]
        )
    ]

    weekly = (
        filtered.groupby("week")["num_orders"]
        .sum()
        .reset_index()
        .sort_values("week")
    )

    st.subheader("Weekly Order Trends")

    st.line_chart(
        weekly,
        x="week",
        y="num_orders",
        color="#38BDF8"
    )

    st.divider()

    category_data = filtered.merge(
        meals,
        on="meal_id",
        how="left"
    )

    left, right = st.columns(2)

    with left:

        st.subheader("Category-wise Demand")

        category_orders = (
            category_data.groupby("category")["num_orders"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            category_orders,
            horizontal=True,
            color="#2DD4BF"
        )

    with right:

        st.subheader("Centre-wise Demand")

        centre_orders = (
            filtered.groupby("center_id")["num_orders"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(
            centre_orders,
            horizontal=True,
            color="#818CF8"
        )

    st.metric(
        "Orders in Selected Period",
        f"{filtered['num_orders'].sum():,.0f}"
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.subheader("Machine Learning Model Evaluation")

    st.write(
        "The following results are from the initial random "
        "train/test split used during model development."
    )

    metrics = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ],
        "MAE": [
            161.33,
            79.97,
            83.96
        ],
        "RMSE": [
            298.33,
            158.17,
            161.39
        ],
        "R2 Score": [
            0.4165,
            0.8360,
            0.8292
        ]
    })

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Best Model",
        "Random Forest"
    )

    c2.metric(
        "Mean Absolute Error",
        "79.97"
    )

    c3.metric(
        "R² Score",
        "0.8360"
    )

    st.subheader("Model Comparison")

    st.write("Mean Absolute Error — lower is better.")

    st.bar_chart(
        metrics.set_index("Model")["MAE"],
        color="#38BDF8"
    )

    st.write("R² Score — higher is better.")

    st.bar_chart(
        metrics.set_index("Model")["R2 Score"],
        color="#2DD4BF"
    )

    st.info(
        "Random Forest was selected because it achieved the "
        "lowest MAE in the initial evaluation. These results "
        "use a random split and should not yet be interpreted "
        "as verified future-week forecasting performance."
    )


# --------------------------------------------------
# PREDICTION STUDIO
# --------------------------------------------------

elif page == "Prediction Studio":

    st.subheader("Demand Prediction Studio")

    st.write(
        "Select a valid meal and fulfilment centre to "
        "forecast demand and estimate potential inventory surplus."
    )

    st.sidebar.subheader("Prediction Configuration")

    center_ids = sorted(train["center_id"].unique())

    selected_center = st.sidebar.selectbox(
        "Fulfilment Centre",
        center_ids
    )

    available_meals = sorted(
        train.loc[
            train["center_id"] == selected_center,
            "meal_id"
        ].unique()
    )

    selected_meal = st.sidebar.selectbox(
        "Meal",
        available_meals
    )

    meal_details = meals.loc[
        meals["meal_id"] == selected_meal
    ].iloc[0]

    center_details = centers.loc[
        centers["center_id"] == selected_center
    ].iloc[0]

    historical = train[
        (train["center_id"] == selected_center) &
        (train["meal_id"] == selected_meal)
    ]

    latest_record = historical.sort_values(
        "week"
    ).iloc[-1]

    st.markdown("#### Selected Meal")

    a, b, c = st.columns(3)

    a.metric("Meal ID", selected_meal)
    b.metric("Category", meal_details["category"])
    c.metric("Cuisine", meal_details["cuisine"])

    st.caption(
        f"Centre {selected_center} | "
        f"{center_details['center_type']} | "
        f"Operating Area: {center_details['op_area']}"
    )

    st.divider()

    st.markdown("#### Business Inputs")

    left, right = st.columns(2)

    with left:

        week = st.number_input(
            "Forecast Week",
            min_value=1,
            max_value=200,
            value=min(
                int(latest_record["week"]) + 1,
                200
            )
        )

        checkout_price = st.number_input(
            "Checkout Price",
            min_value=0.01,
            value=float(latest_record["checkout_price"])
        )

        base_price = st.number_input(
            "Base Price",
            min_value=0.01,
            value=float(latest_record["base_price"])
        )

    with right:

        emailer = st.selectbox(
            "Email Promotion",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        homepage = st.selectbox(
            "Homepage Featured",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        planned_inventory = st.number_input(
            "Planned Inventory (Meals)",
            min_value=0,
            value=1000,
            step=10
        )

    st.divider()

    if st.button(
        "Generate Prediction",
        type="primary",
        use_container_width=True
    ):

        try:

            predicted_demand = predict_demand(
                week=week,
                center_id=selected_center,
                meal_id=selected_meal,
                checkout_price=checkout_price,
                base_price=base_price,
                emailer_for_promotion=emailer,
                homepage_featured=homepage,
                category=meal_details["category"],
                cuisine=meal_details["cuisine"],
                city_code=center_details["city_code"],
                region_code=center_details["region_code"],
                center_type=center_details["center_type"],
                op_area=center_details["op_area"]
            )

            result = estimate_surplus(
                predicted_demand,
                planned_inventory
            )

            st.success("Prediction generated successfully!")

            st.markdown("### Prediction Results")

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Predicted Demand",
                f"{predicted_demand:,} meals"
            )

            p2.metric(
                "Planned Inventory",
                f"{planned_inventory:,} meals"
            )

            p3.metric(
                "Potential Surplus",
                f"{result['potential_surplus']:,} meals"
            )

            st.divider()

            st.markdown("### Surplus Risk Assessment")

            risk = result["risk_level"]

            if risk == "Low":
                st.success(f"Risk Level: {risk}")

            elif risk == "Medium":
                st.warning(f"Risk Level: {risk}")

            else:
                st.error(f"Risk Level: {risk}")

            st.metric(
                "Potential Surplus Percentage",
                f"{result['surplus_percentage']}%"
            )

            st.info(result["recommendation"])

            st.markdown("### Inventory Breakdown")

            chart_data = pd.DataFrame({
                "Inventory Component": [
                    "Predicted Demand",
                    "Potential Surplus"
                ],
                "Meals": [
                    result["predicted_demand"],
                    result["potential_surplus"]
                ]
            })

            st.bar_chart(
                chart_data,
                x="Inventory Component",
                y="Meals",
                color="#38BDF8"
            )

            st.caption(
                "Potential surplus is estimated as planned "
                "inventory above predicted demand. It does "
                "not represent measured discarded food."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Smart Food Waste Intelligence | "
    "Machine Learning • Demand Forecasting • Inventory Analytics"
)

st.caption(
    "Research prototype: surplus estimates are based on "
    "planned inventory and predicted orders, not actual "
    "food disposal records."
)
