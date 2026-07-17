
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from config import *
from components import *

def predictive_page(model_data):
    page_header("ML Revenue Prediction", "Predict future revenue based on order characteristics")
    
    if not model_data:
        st.warning("Model not found! Please run the predictive model script first.")
        return

    model = model_data["model"]
    feature_columns = model_data["feature_columns"]
    metrics = model_data["metrics"]

    section_header("Model Performance Metrics")
    model_metrics(metrics)

    divider()
    section_header("🔮 Predict Order Revenue")

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
        prediction_card(max(0, prediction))
    dashboard_footer()
