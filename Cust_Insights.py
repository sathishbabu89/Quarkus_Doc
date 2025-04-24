import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from huggingface_hub import InferenceClient

# Configuration
QUARKUS_API = st.secrets.get("QUARKUS_API", "http://localhost:8080")
HUGGINGFACE_API_TOKEN = st.secrets.get("HUGGINGFACE_API_TOKEN")  # Add to your secrets
CACHE_TTL = 3600  # 1 hour cache

st.title("Financial Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["Overview", "AI Insights"])

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

def generate_savings_recommendations(financial_data):
    """Generate personalized savings recommendations using HuggingFace LLM"""
    client = InferenceClient(token=HUGGINGFACE_API_TOKEN)
    
    prompt = f"""
    [INST] As a financial advisor, analyze this financial data and provide personalized savings recommendations:
    
    - Monthly Income: €{financial_data.get('totalIncome', 0):,.2f}
    - Monthly Expenses: €{financial_data.get('totalExpenses', 0):,.2f}
    - Current Savings: €{financial_data.get('savings', 0):,.2f}
    - Spending by Category: {financial_data.get('spending_by_category', {})}
    
    Provide:
    1. Three specific savings opportunities based on spending patterns
    2. Recommended savings goals based on income level
    3. Actionable steps to achieve these goals
    4. Potential long-term benefits of these savings
    
    Use bullet points and professional but friendly language. [/INST]
    """
    
    response = client.text_generation(
        prompt,
        model="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=512,
        temperature=0.7
    )
    
    return response

with tab1:
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

with tab2:
    st.subheader("🤖 AI-Powered Savings Recommendations")
    st.markdown("""
    <div style="background-color: #e7f5fe; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <b>Personalized financial advice</b> powered by Mistral-7B AI model
    </div>
    """, unsafe_allow_html=True)
    
    if not HUGGINGFACE_API_TOKEN:
        st.warning("Please configure your HuggingFace API token to enable AI features")
    elif total_summary and spending:
        if st.button("Generate Savings Recommendations", type="primary"):
            with st.spinner("Analyzing your finances with AI..."):
                try:
                    financial_data = {
                        'totalIncome': total_summary.get('totalIncome', 0),
                        'totalExpenses': total_summary.get('totalExpenses', 0),
                        'savings': total_summary.get('savings', 0),
                        'spending_by_category': spending
                    }
                    
                    recommendations = generate_savings_recommendations(financial_data)
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
                        <h4 style="color: #1a5276;">Your Personalized Savings Plan</h4>
                        {recommendations.replace("\n", "<br>")}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed to generate recommendations: {str(e)}")
                    st.info("Please check your Hugging Face API token and internet connection")
        else:
            st.info("Click the button above to generate personalized savings recommendations")
    else:
        st.warning("Please load your financial data first in the Overview tab")

# Sidebar
st.sidebar.title("Options")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.markdown("### About")
st.sidebar.info("""
This dashboard provides an overview of your financial status, 
including income, expenses, and savings. The AI Insights tab
offers personalized recommendations powered by Mistral-7B.
""")
st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
