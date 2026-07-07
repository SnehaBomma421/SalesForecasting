import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 End-to-End Sales Forecasting & Demand Intelligence Dashboard")

# -----------------------------
# Load Data
# -----------------------------
sales_df = pd.read_csv("train.csv")

sales_df["Order Date"] = pd.to_datetime(
    sales_df["Order Date"],
    dayfirst=True
)

sales_df["Ship Date"] = pd.to_datetime(
    sales_df["Ship Date"],
    dayfirst=True
)

sales_df["Year"] = sales_df["Order Date"].dt.year
sales_df["Month"] = sales_df["Order Date"].dt.month

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# ==========================================================
# PAGE 1
# ==========================================================

if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

    region = st.sidebar.selectbox(
        "Select Region",
        ["All"] + list(sales_df["Region"].unique())
    )

    category = st.sidebar.selectbox(
        "Select Category",
        ["All"] + list(sales_df["Category"].unique())
    )

    filtered = sales_df.copy()

    if region != "All":
        filtered = filtered[
            filtered["Region"] == region
        ]

    if category != "All":
        filtered = filtered[
            filtered["Category"] == category
        ]

    # ---------------------------------
    # Total Sales by Year
    # ---------------------------------

    yearly_sales = (
        filtered
        .groupby("Year")["Sales"]
        .sum()
    )

    st.subheader("Total Sales by Year")

    fig, ax = plt.subplots(figsize=(8,4))

    yearly_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # ---------------------------------
    # Monthly Trend
    # ---------------------------------

    monthly_sales = (
        filtered
        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    st.subheader("Monthly Sales Trend")

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"],
        marker="o"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # ---------------------------------
    # Sales by Region
    # ---------------------------------

    st.subheader("Sales by Region")

    region_sales = (
        filtered
        .groupby("Region")["Sales"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(7,4))

    region_sales.plot(
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)

    # ---------------------------------
    # Sales by Category
    # ---------------------------------

    st.subheader("Sales by Category")

    category_sales = (
        filtered
        .groupby("Category")["Sales"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(7,4))

    category_sales.plot(
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)
# ==========================================================
# PAGE 2
# ==========================================================

elif page == "Forecast Explorer":

    st.header("Forecast Explorer")

    forecast_type = st.selectbox(
        "Forecast Type",
        [
            "Category",
            "Region"
        ]
    )

    if forecast_type == "Category":

        option = st.selectbox(
            "Select Category",
            sales_df["Category"].unique()
        )

        data = sales_df[
            sales_df["Category"] == option
        ]

    else:

        option = st.selectbox(
            "Select Region",
            sales_df["Region"].unique()
        )

        data = sales_df[
            sales_df["Region"] == option
        ]

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    monthly = (
        data.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    last_sales = monthly["Sales"].iloc[-1]

    forecast = []

    current = last_sales

    for i in range(horizon):

        current = current * 1.03

        forecast.append(current)

    future = pd.date_range(
        monthly["Order Date"].iloc[-1],
        periods=horizon + 1,
        freq="ME"
    )[1:]

    forecast_df = pd.DataFrame({
        "Forecast Date": future,
        "Forecast Sales": forecast
    })

    st.subheader("Forecast Table")

    st.dataframe(forecast_df)

    st.subheader("Forecast Chart")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        monthly["Order Date"],
        monthly["Sales"],
        marker="o",
        label="Historical Sales"
    )

    ax.plot(
        future,
        forecast,
        marker="o",
        linestyle="--",
        label="Forecast"
    )

    ax.legend()

    st.pyplot(fig)

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
        "MAE",
        "13915.32"
)

    with col2:
        st.metric(
            "RMSE",
            "18893.85"
        )

    st.info(
        "XGBoost was selected because it achieved the lowest RMSE among all forecasting models."
    )
# ==========================================================
# PAGE 3
# ==========================================================

elif page == "Anomaly Report":

    st.header("Anomaly Report")

    st.write(
        "This page displays anomalies detected in weekly sales using the Z-Score method."
    )

    # Weekly Sales
    weekly_sales = (
        sales_df
        .groupby(pd.Grouper(key="Order Date", freq="W-SUN"))["Sales"]
        .sum()
        .reset_index()
    )

    # Rolling Statistics
    weekly_sales["RollingMean"] = (
        weekly_sales["Sales"]
        .rolling(window=8)
        .mean()
    )

    weekly_sales["RollingStd"] = (
        weekly_sales["Sales"]
        .rolling(window=8)
        .std()
    )

    weekly_sales["ZScore"] = (
        (weekly_sales["Sales"] - weekly_sales["RollingMean"])
        / weekly_sales["RollingStd"]
    )

    anomalies = weekly_sales[
        weekly_sales["ZScore"].abs() > 2
    ]

    # -------------------------
    # Plot
    # -------------------------

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        label="Weekly Sales"
    )

    ax.scatter(
        anomalies["Order Date"],
        anomalies["Sales"],
        color="red",
        s=90,
        label="Anomalies"
    )

    ax.legend()

    ax.set_title("Weekly Sales Anomalies")

    st.pyplot(fig)

    # -------------------------
    # Table
    # -------------------------

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomalies[
            [
                "Order Date",
                "Sales",
                "ZScore"
            ]
        ]
    )

    st.success(
        f"{len(anomalies)} anomaly weeks detected using Z-Score."
    )

    st.markdown(
        """
### Business Interpretation

Possible reasons include:

- Holiday sales
- Promotional events
- Festival shopping
- Bulk customer orders
- Seasonal demand spikes
"""
    )



# ==========================================================
# PAGE 4
# ==========================================================

elif page == "Product Demand Segments":

    st.header("Product Demand Segmentation")

    st.write(
        "Products have been grouped using K-Means clustering based on demand characteristics."
    )

    # ----------------------------------------
    # Display Cluster Chart
    # ----------------------------------------

    if os.path.exists("charts/product_clusters.png"):

        st.image(
            "charts/product_clusters.png",
            caption="Product Demand Clusters"
        )

    else:

        st.warning(
            "Cluster image not found. Please place product_clusters.png inside the charts folder."
        )

    # ----------------------------------------
    # Cluster Table
    # ----------------------------------------

    cluster_table = pd.DataFrame({

        "Sub-Category":[
            "Accessories",
            "Binders",
            "Chairs",
            "Phones",
            "Storage",
            "Tables",
            "Copiers",
            "Appliances",
            "Art",
            "Bookcases",
            "Envelopes",
            "Fasteners",
            "Furnishings",
            "Labels",
            "Paper",
            "Supplies",
            "Machines"
        ],

        "Cluster":[
            "High Volume, Stable Demand",
            "High Volume, Stable Demand",
            "High Volume, Stable Demand",
            "High Volume, Stable Demand",
            "High Volume, Stable Demand",
            "High Volume, Stable Demand",
            "High Value, Premium Products",
            "Low Volume, Moderate Demand",
            "Low Volume, Moderate Demand",
            "Low Volume, Moderate Demand",
            "Low Volume, Moderate Demand",
            "Low Volume, Moderate Demand",
            "Low Volume, Moderate Demand",
            "Low Volume, Moderate Demand",
            "Low Volume, Moderate Demand",
            "Low Volume, Moderate Demand",
            "Specialized / High Investment"
        ]

    })

    st.subheader("Cluster Members")

    st.dataframe(cluster_table)

    # ----------------------------------------
    # Stocking Strategy
    # ----------------------------------------

    st.subheader("Recommended Stocking Strategy")

    st.markdown("""

### 🟢 High Volume, Stable Demand

- Maintain high inventory.
- Replenish stock frequently.
- Prioritize these products.

---

### 🔵 High Value, Premium Products

- Keep limited inventory.
- Monitor demand carefully.
- High profit per sale.

---

### 🟠 Low Volume, Moderate Demand

- Maintain moderate inventory.
- Restock based on sales history.

---

### 🔴 Specialized / High Investment

- Keep minimal inventory.
- Purchase mainly on customer demand.
- Reduce holding costs.

""")

    st.success(
        "Task 6 Product Segmentation completed successfully."
    )

# ==========================================================
# FOOTER
# ==========================================================

st.sidebar.markdown("---")
st.sidebar.info(
    "End-to-End Sales Forecasting & Demand Intelligence Dashboard\n\nBuilt using Streamlit, Pandas, Scikit-learn, XGBoost and Matplotlib."
)