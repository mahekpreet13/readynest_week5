"""
Phase 5: Interactive Dashboard
==============================
Streamlit application for e-commerce data visualization and predictive analytics.
"""

import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReadyNest E-Commerce Analytics",
    page_icon="RN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
CLEANED_CSV  = BASE_DIR / "data" / "cleaned" / "ecommerce_cleaned.csv"
MONTHLY_CSV  = BASE_DIR / "data" / "cleaned" / "monthly_summary.csv"
MODEL_PATH   = BASE_DIR / "models" / "revenue_model.pkl"
REPORTS_DIR  = BASE_DIR / "reports"
BACKGROUND_IMAGE = Path(__file__).parent / "predictive-analysis-background.png"
DEEP_DIVE_BACKGROUND = Path(__file__).parent / "predictive-analysis-background.png"
PREDICTIVE_BACKGROUND = Path(__file__).parent / "predictive-analysis-background.png"

CATEGORY_OPTIONS = [
    "Electronics",
    "Fashion",
    "Home",
    "Groceries",
]
CITY_OPTIONS = ["Dhanbad", "Bokaro", "Ranchi", "Giridih", "Deoghar", "Hazaribagh", "Jamshedpur", "Ramgarh", "Dumka", "Palamu"]
PAYMENT_OPTIONS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @keyframes slideInSidebar {
        0% { transform: translateX(-20px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    @keyframes fadeInBrand {
        0% { opacity: 0; transform: scale(0.95); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0, 0, 0, 0.08);
        animation: slideInSidebar 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }
    [data-testid="stSidebar"] * {
        color: #1A1A1A !important;
    }
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stMultiSelect > label,
    [data-testid="stSidebar"] .stNumberInput > label,
    [data-testid="stSidebar"] .stSlider > label {
        color: #333333 !important;
        font-weight: 500;
        transition: transform 0.2s ease-in-out, color 0.2s ease-in-out;
    }
    /* Subtle hover effect for radio choices */
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        transform: translateX(6px);
        color: #000000 !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] section[data-testid="stSidebar"] {
        background: transparent;
    }
    .sidebar-brand {
        background: linear-gradient(135deg, #FFFFFF, #F1F3F5);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        font-size: 16px;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        color: #111111;
        font-weight: 800;
        animation: fadeInBrand 0.8s ease-out 0.2s both;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .sidebar-brand:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .sidebar-brand small {
        display: block;
        font-weight: 600;
        color: #6C757D;
        margin-top: 6px;
        font-size: 13px;
    }
    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1A1A1A, #000000) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.3) !important;
        background: linear-gradient(135deg, #333333, #1A1A1A) !important;
    }
    .kpi-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00C896;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .kpi-title { font-size: 14px; color: #A0A0A0; margin-bottom: 5px; }
    .kpi-value { font-size: 28px; font-weight: bold; color: #FFFFFF; }
    .kpi-card{
    transition:.25s;
}

.kpi-card:hover{
    transform:translateY(-3px);
    box-shadow:0 12px 30px rgba(0,200,150,.25);
}
.chart-container{
background:rgba(20,20,30,.55);
padding:15px;
border-radius:16px;
border:1px solid rgba(255,255,255,.08);
backdrop-filter:blur(8px);
}
</style>
""", unsafe_allow_html=True)


def apply_overview_background():
    if not BACKGROUND_IMAGE.exists():
        return

    image_b64 = base64.b64encode(BACKGROUND_IMAGE.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    linear-gradient(rgba(8,10,20,.90), rgba(8,10,20,.90)),
                    url("data:image/jpeg;base64,{image_b64}");
                background-size: cover;
                background-position: center center;
                background-attachment: fixed;
            }}
            section.main > div {{
                background: transparent;
            }}
            [data-testid="stHeader"] {{
                background: rgba(0, 0, 0, 0);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_deep_dive_background():
    if not DEEP_DIVE_BACKGROUND.exists():
        return

    image_b64 = base64.b64encode(DEEP_DIVE_BACKGROUND.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    linear-gradient(135deg, rgba(3, 8, 24, 0.82), rgba(8, 18, 45, 0.72)),
                    url("data:image/jpeg;base64,{image_b64}");
                background-size: cover;
                background-position: center center;
                background-attachment: fixed;
            }}
            section.main > div {{
                background: transparent;
            }}
            [data-testid="stHeader"] {{
                background: rgba(0, 0, 0, 0);
            }}
            div[data-testid="stPlotlyChart"] {{
                background: rgba(6, 10, 22, 0.36);
                border: 1px solid rgba(130, 220, 255, 0.15);
                border-radius: 18px;
                padding: 12px;
                backdrop-filter: blur(8px);
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_predictive_background():
    if not PREDICTIVE_BACKGROUND.exists():
        return

    image_b64 = base64.b64encode(PREDICTIVE_BACKGROUND.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    linear-gradient(135deg, rgba(7, 4, 28, 0.84), rgba(2, 12, 34, 0.76)),
                    url("data:image/png;base64,{image_b64}");
                background-size: cover;
                background-position: center center;
                background-attachment: fixed;
            }}
            section.main > div {{
                background: transparent;
            }}
            [data-testid="stHeader"] {{
                background: rgba(0, 0, 0, 0);
            }}
            div[data-testid="stMetric"] {{
                background: rgba(7, 10, 22, 0.42);
                border: 1px solid rgba(120, 160, 255, 0.16);
                border-radius: 16px;
                padding: 8px 10px;
                backdrop-filter: blur(8px);
            }}
            div[data-testid="stForm"] {{
                background: rgba(7, 10, 22, 0.32);
                border: 1px solid rgba(120, 160, 255, 0.14);
                border-radius: 18px;
                padding: 16px;
                backdrop-filter: blur(10px);
            }}
            div[data-testid="stPlotlyChart"] {{
                background: rgba(7, 10, 22, 0.34);
                border: 1px solid rgba(120, 160, 255, 0.14);
                border-radius: 18px;
                padding: 12px;
                backdrop-filter: blur(8px);
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CLEANED_CSV)
        monthly = pd.read_csv(MONTHLY_CSV)
        return df, monthly
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_resource
def load_ml_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None

df, df_monthly = load_data()
model_data = load_ml_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        END-TO-END DATA PIPELINE
        & PREDICTIVE ANALYSIS</small>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go To:", [
    "Dashboard Overview",
    "Deep Dive EDA",
    "Predictive Analytics",
    "Business Insights",
])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard Overview":
    apply_overview_background()
    st.title("E-Commerce Sales Overview")

    if df.empty:
        st.warning("No data found! Please run the ETL pipeline first.")
        st.stop()

    # Filters
    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        selected_year = st.selectbox("Select Year", options=["All"] + list(df["order_year"].unique()))
    with col2:
        selected_city = st.selectbox("Select City", options=["All"] + list(df["city"].unique()))

    # Filter data
    filtered_df = df.copy()
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df["order_year"] == selected_year]
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df["city"] == selected_city]

    delivered = filtered_df[filtered_df["status"] == "Delivered"]

    # KPIs
    st.markdown("### Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    tot_rev = delivered["revenue"].sum()
    tot_ord = len(delivered)
    avg_ord = delivered["revenue"].mean()
    top_cat = delivered.groupby("category")["revenue"].sum().idxmax() if not delivered.empty else "N/A"

    with kpi1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Revenue</div><div class="kpi-value">${tot_rev/1e6:.2f}M</div></div>', unsafe_allow_html=True)
    with kpi2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Orders</div><div class="kpi-value">{tot_ord:,}</div></div>', unsafe_allow_html=True)
    with kpi3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Order Value</div><div class="kpi-value">${avg_ord:.2f}</div></div>', unsafe_allow_html=True)
    with kpi4: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Top Category</div><div class="kpi-value">{top_cat}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Charts Row 1
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Revenue Trend")
        if not df_monthly.empty:
            trend = df_monthly.copy()
            if selected_year != "All":
                trend = trend[trend["order_year"] == selected_year]
            trend["period"] = trend["order_year"].astype(str) + "-" + trend["order_month"].astype(str).str.zfill(2)
            fig_trend = px.line(trend, x="period", y="total_revenue", markers=True,
                                color_discrete_sequence=["#00C896"])
            fig_trend.update_layout(xaxis_title="Month", yaxis_title="Revenue ($)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        st.subheader("Payment Method Breakdown")
        pm_counts = filtered_df["payment_method"].value_counts().reset_index()
        pm_counts.columns = ["Payment Method", "Count"]
        fig_pie = px.pie(pm_counts, names="Payment Method", values="Count", hole=0.5,
                         color_discrete_sequence=["#00C896", "#1DB954", "#48CAE4", "#023E8A", "#0077B6"])
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DEEP DIVE EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Deep Dive EDA":
    apply_deep_dive_background()
    st.title("Deep Dive Analysis")

    if df.empty:
        st.warning("No data found!")
        st.stop()

    delivered = df[df["status"] == "Delivered"]

    col1,col2,col3 = st.columns([1,1,1])

    with col1:
        st.subheader("Revenue by Category")
        cat_rev = delivered.groupby("category")["revenue"].sum().reset_index().sort_values("revenue")
        fig_bar = px.bar(cat_rev, x="revenue", y="category", orientation="h",
                         color_discrete_sequence=["#00C896"])
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("City Performance (Top 5)")
        city_rev = delivered.groupby("city")["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(5)
        fig_chan = px.bar(city_rev, x="city", y="revenue",
                          color_discrete_sequence=["#48CAE4"])
        fig_chan.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_chan, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Order Status Distribution")
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_stat = px.pie(status_counts, names="Status", values="Count",
                          color_discrete_sequence=px.colors.sequential.Teal)
        fig_stat.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_stat, use_container_width=True)

    with col4:
        st.subheader("Revenue by Day of Week")
        dow_rev = delivered.groupby("order_dow_name")["revenue"].sum().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ).reset_index()
        fig_dow = px.bar(dow_rev, x="order_dow_name", y="revenue",
                         color_discrete_sequence=["#1DB954"])
        fig_dow.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dow, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictive Analytics":
    apply_predictive_background()
    st.title("ML Revenue Prediction")

    if not model_data:
        st.warning("Model not found! Please run the predictive model script first.")
        st.stop()

    model = model_data["model"]
    feature_columns = model_data["feature_columns"]
    metrics = model_data["metrics"]

    st.markdown("### Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"${metrics['rmse']:,.2f}")
    col2.metric("MAE", f"${metrics['mae']:,.2f}")
    col3.metric("R² Score", f"{metrics['r2']:.4f}")

    st.markdown("---")
    st.subheader("🔮 Predict Order Revenue")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            unit_price = st.number_input("Unit Price ($)", min_value=1.0, value=50.0)
            quantity = st.number_input("Quantity", min_value=1, value=1)
            discount_pct = st.slider("Discount %", 0, 50, 0)
            has_discount = 1 if discount_pct > 0 else 0

        with c2:
            category = st.selectbox("Category", CATEGORY_OPTIONS)
            city = st.selectbox("City", CITY_OPTIONS)
            payment_method = st.selectbox("Payment Method", PAYMENT_OPTIONS)

        with c3:
            order_month = st.slider("Order Month", 1, 12, datetime.now().month)
            order_dow = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
            is_weekend = 1 if order_dow in [5, 6] else 0
            order_quarter = (order_month - 1) // 3 + 1

        submit = st.form_submit_button("Generate Prediction")

    if submit:
        input_row = pd.DataFrame([{
            "unit_price": unit_price,
            "quantity": quantity,
            "discount_pct": discount_pct,
            "order_month": order_month,
            "order_quarter": order_quarter,
            "order_dow": order_dow,
            "is_weekend": is_weekend,
            "has_discount": has_discount,
            "category": category,
            "city": city,
            "payment_method": payment_method,
        }])
        encoded = pd.get_dummies(input_row, columns=["category", "city", "payment_method"], drop_first=False)
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)
        prediction = (model["intercept"] + encoded.to_numpy(dtype=float) @ np.asarray(model["coef"], dtype=float))[0]

        st.success(f"### Predicted Revenue: ${max(0, prediction):,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Business Insights":
    st.title("Business Insights & Recommendations")

    insights_path = REPORTS_DIR / "business_insights.md"
    try:
        with open(insights_path, "r", encoding="utf-8") as f:
            insights_text = f.read().strip()
            st.markdown(
                """
                <div style="padding: 0.85rem 1rem; border-radius: 12px; border: 1px solid rgba(0, 200, 150, 0.25); background: rgba(0, 200, 150, 0.06); margin-bottom: 1rem;">
                    <strong>Quick summary:</strong> Electronics and Clothing drive most revenue, heavy discounts compress margin, and the most practical improvement is tighter discount governance plus loyalty retention for high-value customers.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(insights_text)
    except FileNotFoundError:
        st.info("The business insights report will be available after analysis is complete.")
        st.markdown("""
        ### Placeholder Insights
        1. **Electronics & Clothing** drive the majority of revenue.
        2. **Discounts > 20%** significantly erode profit margins without proportionate volume gains.
        3. **Weekend sales** represent a missed opportunity; targeted weekend promos could boost revenue.
        """)