
import streamlit as st
import pandas as pd
from config import *
from components import *
from plots import *

def dashboard_filters(df):
    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.selectbox("Year", ["All"] + sorted(df.order_year.unique().tolist()))
    with c2:
        city = st.selectbox("City", ["All"] + sorted(df.city.unique().tolist()))
    with c3:
        category = st.selectbox("Category", ["All"] + sorted(df.category.unique().tolist()))
    
    filtered = df.copy()
    if year != "All":
        filtered = filtered[filtered.order_year == year]
    if city != "All":
        filtered = filtered[filtered.city == city]
    if category != "All":
        filtered = filtered[filtered.category == category]
    return filtered

def calculate_kpis(df):
    delivered = df[df.status == "Delivered"]
    revenue = delivered.revenue.sum()
    orders = len(delivered)
    avg_order = delivered.revenue.mean() if len(delivered) > 0 else 0
    top_category = delivered.groupby("category")["revenue"].sum().idxmax() if not delivered.empty else "N/A"
    return {"Revenue": revenue, "Orders": orders, "Average": avg_order, "Category": top_category}

def show_kpis(kpi):
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Revenue", f"${kpi['Revenue']/1e6:.2f}M")
    with c2: kpi_card("Orders", f"{kpi['Orders']:,}")
    with c3: kpi_card("Average Order", f"${kpi['Average']:,.2f}")
    with c4: kpi_card("Top Category", kpi["Category"], icon="🏆")

def dashboard_page(df, monthly_df):
    page_header("Executive Dashboard", "Business performance overview")
    if df.empty:
        empty_state("Dataset not found.")
        return
    filtered = dashboard_filters(df)
    divider()
    section_header("Key Performance Indicators")
    kpi = calculate_kpis(filtered)
    show_kpis(kpi)
    divider()
    
    left, right = st.columns([2,1])
    with left: chart_card("Revenue Trend", revenue_trend(monthly_df))
    with right: chart_card("Payment Distribution", payment_distribution(filtered))
    
    left, right = st.columns(2)
    with left: chart_card("Revenue by Category", revenue_by_category(filtered[filtered.status=="Delivered"]))
    with right: chart_card("Revenue by City", revenue_by_city(filtered[filtered.status=="Delivered"]))
    
    chart_card("Orders by Category", orders_by_category(filtered))
    chart_card("Revenue by Weekday", weekday_revenue(filtered[filtered.status=="Delivered"]))
    
    divider()
    executive_summary("Executive Summary", f'''
Revenue reached ${kpi['Revenue']:,.0f} across {kpi['Orders']:,} completed orders.
The highest performing category is {kpi['Category']}.
Average order value is ${kpi['Average']:,.2f}.
Overall sales are concentrated around high-value product categories, indicating strong customer preference and healthy transaction sizes.
''')
    dashboard_footer()
