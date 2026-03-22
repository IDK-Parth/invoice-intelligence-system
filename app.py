import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from interence.predict_frieght import predict_invoice_flag
from interence.predict_invoice_flag import predict_invoice_flag


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
# Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging
            
This internal analytics portal provides actionable insights for our procurement team, leveraging machine learning to optimize freight costs and flag potentially risky invoices.

- **Forecast frieght costs accurately**
- **Detect risky invoices**
- **Reduce financial risks**
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
    color: white;
}

/* Cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

/* KPI Card */
.kpi-card {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}

/* Risk banners */
.success-banner {
    background-color: #ecfdf5;
    color: #065f46;
    padding: 15px;
    border-radius: 10px;
    font-weight: 600;
}

.danger-banner {
    background-color: #fef2f2;
    color: #991b1b;
    padding: 15px;
    border-radius: 10px;
    font-weight: 600;
}

/* Section titles */
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Vendor Portal")
module = st.sidebar.radio(
    "Navigation",
    ["Freight Cost Prediction", "Invoice Risk Flagging"]
)

# ---------------- MODULE 1 ----------------
if module == "Freight Cost Prediction":

    st.title("Freight Cost Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Details</div>', unsafe_allow_html=True)

        qty = st.number_input("Item Quantity", min_value=0)
        amount = st.number_input("Total Invoice Amount ($)", min_value=0.0)

        predict = st.button("Predict Freight Cost")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if predict:
            # Dummy prediction logic
            prediction = (qty * 2.5) + (amount * 0.05)

            st.markdown(f"""
            <div class="kpi-card">
                <div style="font-size:16px;">Predicted Freight Cost</div>
                <div style="font-size:32px; font-weight:700;">
                    ${prediction:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Business Impact Section
    st.markdown("### Business Impact")

    data = np.random.randn(20).cumsum()
    st.line_chart(data)


# ---------------- MODULE 2 ----------------
if module == "Invoice Risk Flagging":

    st.title("Invoice Risk Flagging")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Input Risk Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        delay = st.number_input("Receiving Delay (days)", min_value=0)
        mismatch = st.number_input("Quantity Mismatch", min_value=0)

    with col2:
        price_var = st.number_input("Price Variance (%)", min_value=0.0)
        vendor_score = st.number_input("Vendor Risk Score", min_value=0)

    with col3:
        invoice_age = st.number_input("Invoice Age (days)", min_value=0)
        freq_issue = st.number_input("Past Issue Frequency", min_value=0)

    check = st.button("Evaluate Risk")

    st.markdown('</div>', unsafe_allow_html=True)

    if check:
        risk_score = (
            delay * 1.2 +
            mismatch * 1.5 +
            price_var * 1.3 +
            vendor_score * 1.1 +
            invoice_age * 0.8 +
            freq_issue * 1.4
        )

        if risk_score > 50:
            st.markdown("""
            <div class="danger-banner">
                ⚠️ Requires Manual Approval
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-banner">
                ✅ Safe for Auto-Approval
            </div>
            """, unsafe_allow_html=True)

    # Business Impact Section
    st.markdown("### Business Impact")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Risk Trend")
        st.line_chart(np.random.randn(30).cumsum())

    with col2:
        st.caption("Approval Rate")
        st.area_chart(np.random.rand(30))