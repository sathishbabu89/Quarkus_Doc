import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from huggingface_hub import InferenceClient

# Configuration
QUARKUS_API = st.secrets.get("QUARKUS_API", "http://localhost:8080")
HUGGINGFACE_API_TOKEN = st.secrets.get("HUGGINGFACE_API_TOKEN", "")
CACHE_TTL = 3600  # 1 hour cache

# Initialize session state variables
if 'assets' not in st.session_state:
    st.session_state.assets = []
if 'liabilities' not in st.session_state:
    st.session_state.liabilities = []
if 'investments' not in st.session_state:
    st.session_state.investments = []
if 'debts' not in st.session_state:
    st.session_state.debts = []
if 'goals' not in st.session_state:
    st.session_state.goals = []

# App title and setup
st.set_page_config(layout="wide", page_title="Advanced Financial Dashboard")
st.title("💰 Advanced Financial Dashboard")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview", "AI Insights", "Budgets", 
    "Goals", "Forecast", "Debt", 
    "Investments", "Net Worth", "Health Check"
])

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

# Tab 1: Overview
with tab1:
    st.header("📊 Financial Overview")
    
    # UI Loading State
    with st.spinner("Loading financial data..."):
        spending = fetch_data(f"{QUARKUS_API}/analysis/spending-by-category")
        total_summary = fetch_data(f"{QUARKUS_API}/analysis/total-summary")

    # Show metrics if data is available
    if total_summary:
        try:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Income", f"€{total_summary.get('totalIncome', 0):,.2f}")
            col2.metric("Total Expenses", f"€{total_summary.get('totalExpenses', 0):,.2f}")
            col3.metric("Savings", f"€{total_summary.get('savings', 0):,.2f}")
            col4.metric("Savings Rate", f"{(total_summary.get('savings', 0)/total_summary.get('totalIncome', 1)*100):.1f}%")
        except Exception as e:
            st.error(f"Error displaying metrics: {e}")

    # Show spending chart if data is available
    if spending:
        try:
            st.subheader("Spending by Category")
            spending_df = pd.DataFrame.from_dict(spending, orient='index', columns=['Amount'])
            fig = px.pie(spending_df, values='Amount', names=spending_df.index, title="Spending Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Monthly Trend")
            # Simulated trend data - in real app you would fetch this
            trend_data = pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar', 'Apr'],
                'Income': [3000, 3200, 3100, 3300],
                'Expenses': [2200, 2400, 2300, 2350]
            })
            fig = px.line(trend_data, x='Month', y=['Income', 'Expenses'], title="Income vs Expenses")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying charts: {e}")

# Tab 2: AI Insights
with tab2:
    st.header("🤖 AI-Powered Financial Insights")
    
    if not HUGGINGFACE_API_TOKEN:
        st.warning("Please configure your HuggingFace API token to enable AI features")
    elif total_summary and spending:
        if st.button("Generate Financial Recommendations", type="primary"):
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
                        <h4 style="color: #1a5276;">Your Personalized Financial Plan</h4>
                        {recommendations.replace("\n", "<br>")}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed to generate recommendations: {str(e)}")
        else:
            st.info("Click the button above to generate personalized financial recommendations")
    else:
        st.warning("Please load your financial data first in the Overview tab")

# Tab 3: Budget Management
with tab3:
    st.header("💰 Budget Management")
    
    if spending:
        # Budget setup by category
        with st.expander("Set Monthly Budgets", expanded=True):
            categories = list(spending.keys())
            budgets = {}
            cols = st.columns(3)
            for i, cat in enumerate(categories):
                with cols[i % 3]:
                    budgets[cat] = st.number_input(
                        f"{cat} Budget", 
                        min_value=0, 
                        value=int(spending.get(cat, 0)*1.2),
                        key=f"budget_{cat}"
                    )
        
        # Budget vs Actual visualization
        budget_df = pd.DataFrame({
            'Category': list(spending.keys()),
            'Spent': list(spending.values()),
            'Budget': [budgets.get(cat, 0) for cat in spending.keys()]
        })
        
        fig = px.bar(budget_df, x='Category', y=['Spent', 'Budget'], 
                    barmode='group', title="Budget vs Actual Spending")
        st.plotly_chart(fig, use_container_width=True)
        
        # Budget alerts
        over_budget = budget_df[budget_df['Spent'] > budget_df['Budget']]
        if not over_budget.empty:
            st.warning(f"⚠️ Over budget in: {', '.join(over_budget['Category'])}")
        else:
            st.success("All categories within budget!")

# Tab 4: Goals
with tab4:
    st.header("🎯 Financial Goals")
    
    # Goal setup form
    with st.expander("Add New Goal"):
        col1, col2 = st.columns(2)
        with col1:
            goal_name = st.text_input("Goal Name", key="goal_name")
            target_amount = st.number_input("Target Amount", min_value=1, key="target_amount")
        with col2:
            target_date = st.date_input("Target Date", key="target_date")
            current_saved = st.number_input("Currently Saved", min_value=0, key="current_saved")
        
        if st.button("Add Goal", key="add_goal"):
            st.session_state.goals.append({
                "name": goal_name,
                "target": target_amount,
                "saved": current_saved,
                "date": target_date
            })
            st.success("Goal added!")
    
    # Display goals
    if st.session_state.goals:
        for i, goal in enumerate(st.session_state.goals):
            with st.container(border=True):
                cols = st.columns([2,1,1,1,1])
                with cols[0]:
                    st.subheader(goal['name'])
                with cols[1]:
                    st.metric("Target", f"€{goal['target']:,.2f}")
                with cols[2]:
                    st.metric("Saved", f"€{goal['saved']:,.2f}")
                with cols[3]:
                    remaining = max(0, goal['target'] - goal['saved'])
                    st.metric("Remaining", f"€{remaining:,.2f}")
                with cols[4]:
                    progress = min(100, (goal['saved']/goal['target'])*100)
                    st.progress(int(progress), text=f"{progress:.1f}%")
    else:
        st.info("No goals set yet. Add your first financial goal above.")

# Tab 5: Cash Flow Forecast
with tab5:
    st.header("🔮 Cash Flow Forecast")
    
    if total_summary:
        # Forecasting parameters
        with st.expander("Forecast Settings"):
            months = st.slider("Projection Period (months)", 1, 24, 6)
            income_growth = st.number_input("Expected Income Growth (% per month)", value=0.5)
            expense_growth = st.number_input("Expected Expense Growth (% per month)", value=0.3)
        
        # Generate forecast
        dates = pd.date_range(datetime.today(), periods=months, freq='M')
        forecast = []
        for i, date in enumerate(dates):
            forecast.append({
                'Month': date.strftime("%b %Y"),
                'Income': total_summary['totalIncome'] * (1 + income_growth/100)**i,
                'Expenses': total_summary['totalExpenses'] * (1 + expense_growth/100)**i,
                'Savings': (total_summary['totalIncome'] * (1 + income_growth/100)**i) - 
                          (total_summary['totalExpenses'] * (1 + expense_growth/100)**i)
            })
        forecast_df = pd.DataFrame(forecast)
        
        # Visualization
        fig = px.line(forecast_df, x='Month', y=['Income', 'Expenses', 'Savings'], 
                     title="Projected Cash Flow")
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        total_savings = forecast_df['Savings'].sum()
        avg_savings = forecast_df['Savings'].mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Projected Total Savings", f"€{total_savings:,.2f}")
        col2.metric("Average Monthly Savings", f"€{avg_savings:,.2f}")
    else:
        st.warning("Please load financial data in the Overview tab first")

# Tab 6: Debt Management
with tab6:
    st.header("💳 Debt Payoff Planner")
    
    # Debt entry form
    with st.expander("Add New Debt"):
        cols = st.columns(4)
        with cols[0]: 
            debt_name = st.text_input("Debt Name", key="debt_name")
        with cols[1]: 
            debt_balance = st.number_input("Current Balance", min_value=0.0, key="debt_balance")
        with cols[2]: 
            debt_rate = st.number_input("Interest Rate (%)", min_value=0.0, key="debt_rate")
        with cols[3]: 
            debt_payment = st.number_input("Monthly Payment", min_value=0.0, key="debt_payment")
        
        if st.button("Add Debt", key="add_debt"):
            st.session_state.debts.append({
                "name": debt_name,
                "balance": debt_balance,
                "rate": debt_rate,
                "payment": debt_payment
            })
            st.success("Debt added!")
    
    # Payoff strategy
    if st.session_state.debts:
        strategy = st.radio("Payoff Strategy", 
                          ["Snowball (smallest balance first)", 
                           "Avalanche (highest interest first)"],
                          horizontal=True)
        
        # Calculate payoff timeline (simplified)
        payoff_data = []
        for debt in st.session_state.debts:
            months = int(debt['balance'] / debt['payment'])
            interest = debt['balance'] * (debt['rate']/100) * (months/12)
            payoff_data.append({
                "Debt": debt['name'],
                "Months": months,
                "Interest": interest,
                "Payment": debt['payment']
            })
        
        # Sort by selected strategy
        if "Avalanche" in strategy:
            payoff_data.sort(key=lambda x: -x['Interest'])
        else:
            payoff_data.sort(key=lambda x: x['balance'])
        
        payoff_df = pd.DataFrame(payoff_data)
        
        # Display results
        st.subheader("Payoff Timeline")
        st.dataframe(payoff_df.style.format({
            'Interest': '€{:.2f}',
            'Payment': '€{:.2f}'
        }), use_container_width=True)
        
        # Visual payoff plan
        fig = px.bar(payoff_df, x='Debt', y='Months', color='Interest',
                    title="Debt Payoff Timeline")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add your debts to create a payoff plan")

# Tab 7: Investment Tracking
with tab7:
    st.header("📈 Investment Portfolio")
    
    # Investment entry form
    with st.expander("Add Investment"):
        cols = st.columns([2,1,1,1,1])
        with cols[0]: 
            ticker = st.text_input("Ticker/Name", key="inv_ticker")
        with cols[1]: 
            shares = st.number_input("Shares", min_value=0.0, key="inv_shares")
        with cols[2]: 
            cost = st.number_input("Cost Basis", min_value=0.0, key="inv_cost")
        with cols[3]: 
            current = st.number_input("Current Price", min_value=0.0, key="inv_current")
        with cols[4]: 
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("Add", key="add_investment"):
                st.session_state.investments.append({
                    "ticker": ticker, 
                    "shares": shares, 
                    "cost": cost, 
                    "current": current
                })
                st.success("Investment added!")
    
    # Portfolio performance
    if st.session_state.investments:
        portfolio_df = pd.DataFrame(st.session_state.investments)
        portfolio_df['Value'] = portfolio_df['shares'] * portfolio_df['current']
        portfolio_df['Cost'] = portfolio_df['shares'] * portfolio_df['cost']
        portfolio_df['Gain'] = portfolio_df['Value'] - portfolio_df['Cost']
        portfolio_df['Gain%'] = (portfolio_df['Gain'] / portfolio_df['Cost']) * 100
        
        # Summary metrics
        total_value = portfolio_df['Value'].sum()
        total_gain = portfolio_df['Gain'].sum()
        total_cost = portfolio_df['Cost'].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio Value", f"€{total_value:,.2f}")
        col2.metric("Total Invested", f"€{total_cost:,.2f}")
        col3.metric("Total Gain/Loss", f"€{total_gain:,.2f}", 
                   f"{total_gain/total_cost*100:.1f}%")
        
        # Allocation pie chart
        st.subheader("Asset Allocation")
        fig = px.pie(portfolio_df, values='Value', names='ticker', 
                    title="Portfolio Composition")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.subheader("Investment Performance")
        st.dataframe(portfolio_df.style.format({
            'Value': '€{:.2f}',
            'Cost': '€{:.2f}',
            'Gain': '€{:.2f}',
            'Gain%': '{:.1f}%'
        }), use_container_width=True)
    else:
        st.info("Add your investments to track performance")

# Tab 8: Net Worth Tracker
with tab8:
    st.header("🏦 Net Worth Tracker")
    
    # Asset/Liability input
    with st.expander("Add Assets"):
        new_asset = st.text_input("Asset Description", key="asset_desc")
        asset_value = st.number_input("Value", min_value=0, key="asset_value")
        if st.button("Add Asset", key="add_asset"):
            st.session_state.assets.append({
                "description": new_asset, 
                "value": asset_value
            })
    
    with st.expander("Add Liabilities"):
        new_liability = st.text_input("Liability Description", key="liability_desc")
        liability_value = st.number_input("Amount Owed", min_value=0, key="liability_value")
        if st.button("Add Liability", key="add_liability"):
            st.session_state.liabilities.append({
                "description": new_liability, 
                "value": liability_value
            })
    
    # Net worth calculation
    total_assets = sum(a['value'] for a in st.session_state.assets)
    total_liabilities = sum(l['value'] for l in st.session_state.liabilities)
    net_worth = total_assets - total_liabilities
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assets", f"€{total_assets:,.2f}")
    col2.metric("Total Liabilities", f"€{total_liabilities:,.2f}")
    col3.metric("Net Worth", f"€{net_worth:,.2f}", 
               delta_color="inverse" if net_worth < 0 else "normal")
    
    # Asset/Liability breakdown
    if st.session_state.assets or st.session_state.liabilities:
        st.subheader("Breakdown")
        
        # Assets visualization
        if st.session_state.assets:
            assets_df = pd.DataFrame(st.session_state.assets)
            fig = px.pie(assets_df, values='value', names='description', 
                         title="Asset Composition")
            st.plotly_chart(fig, use_container_width=True)
        
        # Liabilities visualization
        if st.session_state.liabilities:
            liabilities_df = pd.DataFrame(st.session_state.liabilities)
            fig = px.bar(liabilities_df, x='description', y='value', 
                        title="Liabilities Breakdown")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add your assets and liabilities to calculate net worth")

# Tab 9: Financial Health Check
with tab9:
    st.header("❤️ Financial Health Check")
    
    if total_summary:
        # Calculate key ratios
        savings_rate = (total_summary['savings'] / total_summary['totalIncome']) * 100
        expense_ratio = (total_summary['totalExpenses'] / total_summary['totalIncome']) * 100
        
        # Health metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Savings Rate", f"{savings_rate:.1f}%", 
                  "Good" if savings_rate > 20 else "Needs Improvement")
        col2.metric("Expense Ratio", f"{expense_ratio:.1f}%", 
                  "Healthy" if expense_ratio < 80 else "High")
        
        # Emergency fund check
        monthly_expenses = total_summary['totalExpenses']
        emergency_fund = st.number_input("Emergency Fund Amount", 
                                       value=3*monthly_expenses,
                                       key="emergency_fund")
        col3.metric("Emergency Fund Coverage", 
                   f"{emergency_fund/monthly_expenses:.1f} months",
                   "6+ months recommended")
        
        # Debt-to-income ratio
        if st.session_state.debts:
            total_debt_payments = sum(d['payment'] for d in st.session_state.debts)
            dti_ratio = (total_debt_payments / total_summary['totalIncome']) * 100
            st.metric("Debt-to-Income Ratio", f"{dti_ratio:.1f}%",
                     "Good" if dti_ratio < 35 else "High")
        
        # Financial health score (simplified)
        health_score = min(100, savings_rate*2 + (100-expense_ratio))
        if st.session_state.debts:
            health_score -= min(30, dti_ratio)
        
        st.progress(int(health_score))
        st.subheader(f"Financial Health Score: {int(health_score)}/100")
        
        # Recommendations
        if health_score < 60:
            st.error("🚨 Financial Health Needs Attention")
            st.markdown("""
            - Reduce discretionary spending
            - Increase income streams if possible
            - Focus on paying down high-interest debt
            """)
        elif health_score < 80:
            st.warning("⚠️ Moderate Financial Health")
            st.markdown("""
            - Continue current savings habits
            - Optimize expense categories
            - Consider additional investments
            """)
        else:
            st.success("✅ Excellent Financial Health!")
            st.markdown("""
            - Maintain current financial practices
            - Explore wealth-building opportunities
            - Consider professional financial advice
            """)
    else:
        st.warning("Please load financial data in the Overview tab first")

# Sidebar
st.sidebar.title("Options")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("### About")
st.sidebar.info("""
This comprehensive financial dashboard helps you:
- Track income, expenses, and savings
- Plan budgets and financial goals
- Manage debts and investments
- Monitor your overall financial health
""")

st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Add some custom CSS for better styling
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }
    .stProgress > div > div > div {
        background-color: #1a5276;
    }
</style>
""", unsafe_allow_html=True)
