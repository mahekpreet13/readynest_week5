
import streamlit as st
import pandas as pd
from config import *
from components import *
from plots import *
from pages.overview import dashboard_filters

def deep_dive_page(df):
    page_header("Deep Dive Analytics", "Explore operational performance and revenue drivers")
    if df.empty:
        empty_state("Dataset not found.")
        return
    
    filtered = dashboard_filters(df)
    delivered = filtered[filtered["status"] == "Delivered"]
    
    divider()
    section_header("Sales Performance", "Revenue trends and category analysis")
    
    left, right = st.columns(2)
    with left: chart_card("Revenue by Category", revenue_by_category(delivered))
    with right: chart_card("Monthly Revenue Comparison", monthly_comparison(delivered))
    
    left, right = st.columns(2)
    with left: chart_card("Quarterly Revenue", quarterly_revenue(delivered))
    with right: chart_card("Revenue by Weekday", weekday_revenue(delivered))
    
    divider()
    section_header("Customer & Order Analysis", "Customer behaviour and order quality")
    
    left, right = st.columns(2)
    with left: chart_card("Order Status Distribution", order_status_distribution(filtered))
    with right: chart_card("Weekend vs Weekday Sales", weekend_sales(delivered))
    
    divider()
    section_header("Geographical Performance", "Revenue contribution by city")
    
    left, right = st.columns(2)
    with left: chart_card("Top Cities", top_cities(delivered))
    with right: chart_card("Revenue by City", revenue_by_city(delivered))
    
    divider()
    section_header("Discount Analysis", "Understanding pricing impact")
    
    left, right = st.columns(2)
    with left: chart_card("Discount Distribution", discount_distribution(filtered))
    with right: chart_card("Revenue vs Discount", revenue_discount_scatter(delivered))
    
    divider()
    section_header("Category Insights", "Revenue concentration")
    
    left, right = st.columns(2)
    with left: chart_card("Category Revenue Treemap", category_share(delivered))
    with right: chart_card("Revenue Distribution", revenue_boxplot(delivered))
    
    divider()
    section_header("Payment Behaviour", "Revenue by payment method")
    chart_card("Payment Method Revenue", payment_revenue(delivered))
    
    divider()
    
    total_revenue = delivered["revenue"].sum()
    avg_order = delivered["revenue"].mean() if not delivered.empty else 0
    top_city = delivered.groupby("city")["revenue"].sum().idxmax() if not delivered.empty else "N/A"
    top_category = delivered.groupby("category")["revenue"].sum().idxmax() if not delivered.empty else "N/A"
    
    executive_summary("Key Findings", f'''
• Total delivered revenue reached ${total_revenue:,.0f}
• Highest-performing city: **{top_city}**
• Best-selling category: **{top_category}**
• Average order value: **${avg_order:,.2f}**
• Revenue is concentrated in a few high-performing categories.
• Discounted orders should be monitored to ensure promotional spending continues generating profitable growth.
• Geographic expansion should prioritize cities already demonstrating strong purchasing behaviour.
''')
    dashboard_footer()
