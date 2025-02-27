import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import re
import numpy as np
from datetime import datetime
from io import BytesIO
import plotly.io as pio
from fpdf import FPDF
from PIL import Image
import base64
import random
import sys
import traceback
import hashlib

# Enhanced Security Configuration
AUTH_CONFIG = {
    "username": st.secrets["general"]["USER"],
    "password": hashlib.sha256(st.secrets["general"]["PASS"].encode()).hexdigest(),
    "max_attempts": 3  # Add login attempt limitation
}
# Configure Google Gemini API
GOOGLE_API_KEY = st.secrets["general"]["API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Add animated background for login */
    .login-bg {position: fixed; top: 0; left: 0; right: 0; bottom: 0; 
              background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
              background-size: 400% 400%; animation: gradient 15s ease infinite;}
    @keyframes gradient {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
    .login-container {position: relative; z-index: 1;}
    /* Add responsive design */
    @media (max-width: 600px) {.swot-grid {grid-template-columns: 1fr;}}
    .login-title {text-align: center; margin-bottom: 2rem; color: #2c3e50;}
    .login-button {width: 100%; margin-top: 1rem;}
    .main {background-color:rgba(248, 249, 250, 0.45);}
    .stButton>button {border-radius: 20px; padding: 10px 25px;}
    .stSelectbox, .stMultiselect, .stSlider {padding: 5px;}
    .report-view {margin: 20px 0; padding: 25px; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .dashboard-header {background: linear-gradient(45deg,rgba(76, 175, 79, 0.56), #45a049); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;}
    .kpi-card {background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;}
    .swot-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;}
    .insight-card {padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

def check_authentication(username, password):
    """Enhanced authentication with attempt tracking"""
    st.session_state.setdefault('login_attempts', 0)
    
    if st.session_state.login_attempts >= AUTH_CONFIG["max_attempts"]:
        st.error("Too many failed attempts. Please try again later.")
        return False
        
    hashed_input = hashlib.sha256(password.encode()).hexdigest()
    if (username == AUTH_CONFIG["username"] and 
        hashed_input == AUTH_CONFIG["password"]):
        st.session_state.login_attempts = 0
        return True
    
    st.session_state.login_attempts += 1
    return False

def render_login_page():
    """Enhanced login page with security features"""
    st.markdown("<div class='login-bg'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown("<h2 class='login-title'>🔒 Enterprise Analytics Login</h2>", unsafe_allow_html=True)
        
        with st.form("auth_form"):
            username = st.text_input("Username", help="Contact admin if you forgot credentials")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if check_authentication(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful! Redirecting...")
                    st.balloons()
                    st.rerun()
                else:
                    remaining = AUTH_CONFIG["max_attempts"] - st.session_state.login_attempts
                    st.error(f"Invalid credentials. {remaining} attempts remaining")
        st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    return pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)

def handle_missing_data(df, strategy):
    strategies = {
        "Drop Rows": df.dropna,
        "Fill with Mean": lambda: df.fillna(df.mean(numeric_only=True)),
        "Fill with Median": lambda: df.fillna(df.select_dtypes(include=np.number).median())
    }
    return strategies.get(strategy, lambda: df)()

# Data processing functions remain unchanged

def generate_dynamic_kpis(df):
    kpis = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) > 0:
        if 'revenue' in numeric_cols:
            kpis['Revenue Growth'] = f"{df['revenue'].pct_change().mean() * 100:.1f}%"
        if 'customers' in df.columns:
            kpis['Customer Retention'] = f"{df['customers'].iloc[-1] / df['customers'].iloc[0] * 100:.1f}%"
        if 'cost' in numeric_cols and 'revenue' in numeric_cols:
            kpis['Profit Margin'] = f"{(df['revenue'].sum() - df['cost'].sum()) / df['revenue'].sum() * 100:.1f}%"
        
        if len(kpis) == 0:
            kpis[f"Avg {numeric_cols[0]}"] = f"{df[numeric_cols[0]].mean():.2f}"
            if len(numeric_cols) > 1:
                kpis[f"Total {numeric_cols[1]}"] = f"{df[numeric_cols[1]].sum():,.0f}"
    
    return kpis

def generate_ai_analysis(df, industry, goal):
    """Add error handling for API calls"""
    try:
        data_sample = df.head(1000).to_dict(orient='records')
        prompt = f"""As a Head senior {industry} analyst, create interactive figures understandable and simple but not vague with Plotly express wrapped in ```python blocks for {goal} using this data:
    {data_sample}
    
    Requirements:
    1. Use appropriate chart types (line, bar, scatter, other)
    2. Use EXACTLY these variable names: fig1, fig2, fig3
    3. Only use columns that exist in the data: {list(df.columns)}
    4. Add meaningful titles and axis labels
    5. Provide a brief insight after each chart/visualization
    6. Include hover interactions
    7. Generate SWOT analysis with clear Strengths, Weaknesses, Opportunities, Threats
    8. Create Few strategic recommendations
    9. Provide 5 deep insights with explanations
    
    Format response EXACTLY like:
    
    ### VISUALIZATIONS ###
    ```python
    # KPI 1 Visualization
    fig1 = px.line(df, x='valid_column_name', y='valid_column_name', title='Meaningful Title')
    ```
    Insight: [Your insight here]

    ```python
    # KPI 2 Visualization 
    fig2 = px.bar(df, x='valid_column_name', y='valid_column_name', title='Meaningful Title')
    ```
    Insight: [Your insight here]
    
    ### SWOT ANALYSIS ###
    - Strengths:
      * Strength 1
      * Strength 2
    - Weaknesses:
      * Weakness 1
      * Weakness 2
    - Opportunities:
      * Opportunity 1
      * Opportunity 2
    - Threats:
      * Threat 1
      * Threat 2
    
    ### RECOMMENDATIONS ###
     1. Recommendation 1
     2. Recommendation 2
     3. Recommendation 3
    
    ### DEEP INSIGHTS ###
     - Insight 1: Explanation
     - Insight 2: Explanation
     - Insight 3: Explanation
     - Insight 4: Explanation
     - Insight 5: Explanation"""
        
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"AI Analysis failed: {str(e)}")
        return ""

def parse_ai_response(response):
    """More robust parsing with error handling"""
    parsed_data = {
        'figures': [],
        'swot': {'Strengths': [], 'Weaknesses': [], 'Opportunities': [], 'Threats': []},
        'recommendations': [],
        'insights': []
    }
    
    try:
        # Enhanced regex patterns with fallbacks
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL) or []
        
        # SWOT parsing with improved pattern matching
        swot_section = re.search(r'### SWOT ANALYSIS ###(.*?)(###|$)', response, re.DOTALL)
        if swot_section:
            for category in parsed_data['swot']:
                items = re.findall(rf'{category}:\s*([\s\S]*?)(?=\n\s*-|\Z)', swot_section.group(1))
                if items:
                    parsed_data['swot'][category] = [item.strip() for item in items[0].split('*') if item.strip()]
        
        # Recommendations with alternative numbering formats
        rec_section = re.search(r'### RECOMMENDATIONS ###(.*?)(###|$)', response, re.DOTALL)
        if rec_section:
            parsed_data['recommendations'] = [re.sub(r'^\d+\.?\s*', '', line).strip() 
                                              for line in rec_section.group(1).split('\n') if line.strip()]
        
        # Insights with flexible formatting
        insights_section = re.search(r'### DEEP INSIGHTS ###(.*?)(###|$)', response, re.DOTALL)
        if insights_section:
            parsed_data['insights'] = [line.strip().split(':', 1)[1].strip() 
                                       for line in insights_section.group(1).split('\n') if ':' in line]
            
    except Exception as e:
        st.error(f"Error parsing AI response: {str(e)}")
        
    return code_blocks, parsed_data['swot'], parsed_data['recommendations'], parsed_data['insights']

def safe_execute_code(code, env):
    try:
        exec(code, env)
        return True, None
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        tb = traceback.format_exc()
        return False, f"{error_type}: {error_message}\n\nTraceback:\n{tb}"

def generate_pdf_report():
    """Stub for PDF generation"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Enterprise Analytics Report", ln=1, align="C")
    # Add additional report details as needed
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    st.download_button("Download PDF", pdf_output.getvalue(), "report.pdf", "application/pdf")

def render_dashboard():
    """Enhanced dashboard with responsive design"""
    # Add session timeout warning
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.now()
    else:
        inactive_time = datetime.now() - st.session_state.last_activity
        if inactive_time.seconds > 1800:  # 30 minute timeout
            st.warning("Session timed out due to inactivity")
            st.session_state.authenticated = False
            st.rerun()
    
    st.markdown("<div class='dashboard-header'><h1>Interactive Analytics Dashboard</h1></div>", unsafe_allow_html=True)
    
    # Dynamic KPIs
    if 'processed_data' in st.session_state:
        df = st.session_state.processed_data
        kpis = generate_dynamic_kpis(df)
        if len(kpis) > 0:
            cols = st.columns(min(len(kpis), 4))
            for i, (kpi, value) in enumerate(kpis.items()):
                if i < len(cols):
                    cols[i].metric(kpi, value)

    # Visualization Tabs
    tab1, tab2 = st.tabs(["Primary Analysis", "Deep Insights"])
    
    with tab1:
        if 'dashboard_data' in st.session_state and 'figures' in st.session_state.dashboard_data:
            for idx, fig in enumerate(st.session_state.dashboard_data['figures'], 1):
                try:
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying visualization {idx}: {str(e)}")
    
    with tab2:
        if 'dashboard_data' in st.session_state and 'insights' in st.session_state.dashboard_data:
            for insight in st.session_state.dashboard_data['insights']:
                st.markdown(f"""
                <div class="insight-card">
                    <h4>🔍 Insight</h4>
                    <p>{insight}</p>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
    
    # SWOT Analysis
    if 'dashboard_data' in st.session_state and 'swot' in st.session_state.dashboard_data:
        st.markdown("## 📊 Strategic Assessment")
        with st.expander("SWOT Analysis", expanded=True):
            cols = st.columns(4)
            for i, (category, items) in enumerate(st.session_state.dashboard_data['swot'].items()):
                with cols[i]:
                    st.subheader(category)
                    for item in items:
                        st.markdown(f"▸ {item}")
    
    # Recommendations
    if 'dashboard_data' in st.session_state and 'recommendations' in st.session_state.dashboard_data:
        st.markdown("## 🎯 Strategic Recommendations")
        for i, rec in enumerate(st.session_state.dashboard_data['recommendations'], 1):
            st.markdown(f"{i}. {rec}")

    # Sidebar enhancements for export and help
    with st.sidebar.expander("📤 Export Results"):
        if st.button("Export Report as PDF"):
            generate_pdf_report()
            
    with st.sidebar.expander("ℹ️ Help"):
        st.info("""
        - Use the main panel for data analysis
        - Click charts to interact
        - Export results using sidebar tools
        """)
    
    # Navigation
    if st.button("← Return to Main Analysis"):
        st.session_state.current_page = 'main'
        st.rerun()

def main_application():
    """Enhanced main app with proper session handling"""
    st.session_state.setdefault('current_page', 'main')
    
    # Persistent sidebar with user info and settings
    with st.sidebar:
        st.write(f"👤 Welcome, {st.session_state.username}")
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                if key not in ['authenticated', 'login_attempts']:
                    del st.session_state[key]
            st.session_state.authenticated = False
            st.rerun()
        
        st.write("---")
        st.write("🔧 Analysis Settings")
        industry = st.selectbox("Industry", ["Retail", "Finance", "Healthcare", "Manufacturing", "Technology"])
        goal = st.selectbox("Goal", ["Revenue Optimization", "Cost Reduction", "Customer Insights", "Operational Efficiency"])
    
    # Main content
    if st.session_state['current_page'] == 'dashboard':
        render_dashboard()
        return

    st.markdown("<div class='dashboard-header'><h1>📈 Enterprise Analytics Suite</h1></div>", unsafe_allow_html=True)
    
    with st.expander("🚀 Quick Start Guide", expanded=True):
        cols = st.columns(3)
        cols[0].info("1. Upload your dataset")
        cols[1].warning("2. Configure settings")
        cols[2].success("3. Generate insights")
    
    with st.form("analysis_form"):
        uploaded_file = st.file_uploader("Upload Dataset (CSV/XLSX)", type=["csv", "xlsx"])
        # Note: Industry and goal are also set in the sidebar. You can remove these from here if desired.
        if st.form_submit_button("Analyze Data"):
            if uploaded_file:
                df = load_data(uploaded_file)
                df = handle_missing_data(df, "Fill with Median")
                st.session_state.processed_data = df
                
                with st.spinner("Generating AI-powered insights..."):
                    analysis_report = generate_ai_analysis(df, industry, goal)
                    code_blocks, swot, recommendations, insights = parse_ai_response(analysis_report)
                    
                    # Execute visualization code
                    figures = []
                    exec_env = {
                        'df': df,
                        'px': px,
                        'np': np,
                        'pd': pd,
                        'fig1': None,
                        'fig2': None,
                        'fig3': None
                    }
                    
                    for code in code_blocks:
                        success, error = safe_execute_code(code, exec_env)
                        if not success:
                            st.error(f"Code execution error:\n{error}")
                        else:
                            # Collect figures explicitly
                            new_figures = [
                                exec_env.get('fig1'),
                                exec_env.get('fig2'),
                                exec_env.get('fig3')
                            ]
                            figures.extend([fig for fig in new_figures if fig is not None])
                    
                    st.session_state.dashboard_data = {
                        'figures': figures,
                        'swot': swot,
                        'recommendations': recommendations,
                        'insights': insights
                    }
                
                st.session_state.current_page = 'dashboard'
                st.rerun()
            else:
                st.error("Please upload a valid dataset")

def main():
    """Enhanced main function with session cleanup"""
    st.session_state.setdefault('authenticated', False)
        
    if not st.session_state.authenticated:
        render_login_page()
    else:
        main_application()

if __name__ == "__main__":
    main()
