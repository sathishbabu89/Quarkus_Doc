import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from huggingface_hub import InferenceClient
from gtts import gTTS
from fpdf import FPDF
import tempfile

# Configuration
QUARKUS_API = st.secrets.get("QUARKUS_API", "http://localhost:8080")
HUGGINGFACE_API_TOKEN = st.secrets.get("HUGGINGFACE_API_TOKEN", "")
CACHE_TTL = 3600

# Title
st.title("Financial Dashboard")

# Cache
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

# Hugging Face LLM call
def generate_savings_recommendation(total_summary, spending, language="English"):
    total_income = total_summary.get('totalIncome', 0)
    total_expenses = total_summary.get('totalExpenses', 0)
    savings = total_summary.get('savings', 0)

    top_spending = sorted(spending.items(), key=lambda x: x[1], reverse=True)[:3]
    top_categories = [f"{k} (€{v:,.2f})" for k, v in top_spending]

    prompt = f"""
    [INST] You are a financial advisor AI providing advice in {language}. Based on the following:
    - Total Income: €{total_income:,.2f}
    - Total Expenses: €{total_expenses:,.2f}
    - Savings: €{savings:,.2f}
    - Top Spending Categories: {', '.join(top_categories)}

    Provide 3 personalized savings recommendations in {language}. Keep it professional and user-friendly. [/INST]
    """

    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=HUGGINGFACE_API_TOKEN
    )
    response = client.text_generation(prompt, max_new_tokens=300, temperature=0.5)
    return response.strip()

# PDF Generation
def generate_pdf_report(content: str, language: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("AI Savings Recommendations")
    pdf.multi_cell(0, 10, f"AI-Powered Savings Insights ({language})\n\n{content}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# Voice Generation
def generate_audio_from_text(text: str, language_code="en"):
    tts = gTTS(text=text, lang=language_code)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

LANGUAGE_CODE_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de"
}

# Load data
with st.spinner("Loading financial data..."):
    spending = fetch_data(f"{QUARKUS_API}/analysis/spending-by-category")
    total_summary = fetch_data(f"{QUARKUS_API}/analysis/total-summary")

# Tabs
tab1, tab2, tab3 = st.tabs(["💰 Overview", "📈 Spending Chart", "🤖 AI-Powered Insights"])

# Overview
with tab1:
    if total_summary:
        try:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Income", f"€{total_summary.get('totalIncome', 0):,.2f}")
            col2.metric("Total Expenses", f"€{total_summary.get('totalExpenses', 0):,.2f}")
            col3.metric("Savings", f"€{total_summary.get('savings', 0):,.2f}")
        except Exception as e:
            st.error(f"Error displaying metrics: {e}")

# Spending chart
with tab2:
    if spending:
        try:
            st.subheader("Spending by Category")
            spending_df = pd.DataFrame.from_dict(spending, orient='index', columns=['Amount'])
            st.bar_chart(spending_df)
        except Exception as e:
            st.error(f"Error displaying spending chart: {e}")

# AI-powered insights tab
with tab3:
    st.subheader("🤖 Personalized Savings Insights")
    st.markdown("""
    <div style="background-color: #e8f6f3; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <b>AI-generated recommendations</b> using Mistral-7B to help optimize your savings.
    </div>
    """, unsafe_allow_html=True)

    language = st.selectbox("🌍 Select Language", ["English", "Spanish", "French", "German"])

    if st.button("Generate Savings Tips", type="primary"):
        if total_summary and spending:
            with st.spinner(f"Generating insights in {language}..."):
                try:
                    insights = generate_savings_recommendation(total_summary, spending, language)

                    # Avatar-style assistant display
                    st.markdown(f"""
                    <div style="display: flex; align-items: flex-start; gap: 15px; margin-top: 20px; background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png" width="60" style="border-radius: 50%;">
                        <div style="flex-grow: 1;">
                            <h4 style="margin-bottom: 10px; color: #1f618d;">FinBot 💼</h4>
                            <div style="background-color: #e8f8f5; padding: 15px; border-radius: 8px; line-height: 1.6; color: #154360;">
                                {insights.replace("\n", "<br>")}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # PDF Download
                    pdf_path = generate_pdf_report(insights, language)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📄 Download Recommendations as PDF",
                            data=f.read(),
                            file_name=f"savings_insights_{language.lower()}.pdf",
                            mime="application/pdf"
                        )

                    # Voice Playback
                    audio_path = generate_audio_from_text(insights, LANGUAGE_CODE_MAP.get(language, "en"))
                    with open(audio_path, "rb") as audio_file:
                        st.markdown("**🎧 FinBot can read this out loud for you:**")
                        st.audio(audio_file.read(), format="audio/mp3")

                except Exception as e:
                    st.error(f"Failed to generate insights: {e}")
        else:
            st.warning("Missing data for insights. Please refresh the dashboard.")

# Sidebar
st.sidebar.title("Options")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.markdown("### About")
st.sidebar.info("""
This dashboard provides an overview of your financial status, 
including income, expenses, and savings. The AI tab offers 
tailored insights, voice guidance, and downloadable PDFs.
""")
st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
