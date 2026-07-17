
import streamlit as st
from pathlib import Path
from config import *
from components import *

def business_insights_page(report_path):
    page_header("Business Insights", "Executive Summary & Strategic Recommendations")
    divider()
    
    st.markdown("### Executive Summary\n\nThis report summarizes the major findings discovered during exploratory data analysis and predictive modelling.")
    
    c1, c2 = st.columns(2)
    with c1:
        executive_summary("Revenue Drivers", "Electronics and Fashion consistently generate the highest revenue and should remain the primary investment categories.", "💰")
        executive_summary("Customer Behaviour", "Customers purchasing multiple items have significantly higher average order values.", "👥")
    with c2:
        executive_summary("Operational Risk", "High discount percentages reduce profitability without proportionally increasing revenue.", "⚠️")
        executive_summary("Growth Opportunity", "Weekend promotions and targeted city campaigns represent the strongest opportunities for increasing sales.", "🚀")
    
    divider()
    section_header("Strategic Priorities")
    
    col1, col2, col3 = st.columns(3)
    with col1: info_card("Highest Revenue Category", "Electronics", COLORS["primary"])
    with col2: info_card("Highest Revenue City", "Ranchi", COLORS["success"])
    with col3: info_card("Recommended Discount", "< 15%", COLORS["warning"])
    
    divider()
    section_header("Recommended Business Actions")
    
    recommendations = [
        ("🎯 Focus Marketing Budget", "Increase investment in Electronics and Fashion campaigns where ROI is consistently highest."),
        ("💳 Improve Payment Experience", "Encourage digital payment methods through cashback and loyalty rewards."),
        ("🏙 Expand High-Performing Cities", "Increase inventory and advertising in cities demonstrating sustained revenue growth."),
        ("📉 Optimize Discounts", "Avoid discounts greater than 20%. Historical analysis shows limited revenue improvement."),
        ("🛍 Increase Weekend Sales", "Launch weekend flash sales and targeted campaigns to improve conversion rates.")
    ]
    
    for title, text in recommendations:
        executive_summary(title, text)
        
    divider()
    section_header("Detailed Business Report")
    
    report = Path(report_path)
    if report.exists():
        with st.expander("View Complete Report", expanded=False):
            st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("Business report not found.")
        
    divider()
    st.success("### Final Recommendation\n\nContinue prioritizing high-performing product categories while reducing unnecessary discounting.\n\nExpand marketing efforts in strong revenue-generating cities and leverage predictive analytics to support pricing and inventory decisions.")
    dashboard_footer()
