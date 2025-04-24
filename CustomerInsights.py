import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Configuration (could be moved to environment variables)
QUARKUS_API = st.secrets.get("QUARKUS_API", "http://localhost:8080")
CACHE_TTL = 3600  # 1 hour cache

st.title("Financial Dashboard")

@st.cache_data(ttl=CACHE_TTL)
def fetch_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None
    except ValueError as e:
        st.error(f"Error decoding JSON: {e}")
        return None

# UI Loading State
with st.spinner("Loading financial data..."):
    spending = fetch_data(f"{QUARKUS_API}/analysis/spending-by-category")
    total_summary = fetch_data(f"{QUARKUS_API}/analysis/total-summary")

# Show metrics if data is available
if total_summary:
    try:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Income", f"€{total_summary.get('totalIncome', 0):,.2f}")
        col2.metric("Total Expenses", f"€{total_summary.get('totalExpenses', 0):,.2f}")
        col3.metric("Savings", f"€{total_summary.get('savings', 0):,.2f}")
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")

# Show spending chart if data is available
if spending:
    try:
        st.subheader("Spending by Category")
        spending_df = pd.DataFrame.from_dict(spending, orient='index', columns=['Amount'])
        st.bar_chart(spending_df)
    except Exception as e:
        st.error(f"Error displaying spending chart: {e}")

# Sidebar
st.sidebar.title("Options")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.markdown("### About")
st.sidebar.info("""
This dashboard provides an overview of your financial status, 
including income, expenses, and savings. The spending chart 
categorizes your expenses to help you understand where your 
money is going.
""")
st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
