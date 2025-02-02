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

# Configure Google Gemini API
GOOGLE_API_KEY = st.secrets["general"]["API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_data
def load_data(file):
    return pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)

def create_pdf_report(content, images):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.set_text_color(33, 37, 41)
            self.cell(0, 10, 'Business Intelligence Report', 0, 1, 'C')
            self.ln(5)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Report metadata
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 10, 'Business Analysis Report', 0, 1, 'C')
    
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(108, 117, 125)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(10)
    
    # Main content
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(33, 37, 41)
    pdf.multi_cell(0, 8, content)
    pdf.ln(5)
    
    # Image handling with high resolution
    for i, img_bytes in enumerate(images):
        pdf.add_page()
        img_bytes.seek(0)
        
        # Open and convert image
        img = Image.open(img_bytes).convert("RGB")
        
        # Calculate dimensions for 300 DPI
        target_width_mm = 180
        mm_to_inch = 1/25.4
        dpi = 300
        target_width_px = int(target_width_mm * mm_to_inch * dpi)
        aspect = img.height / img.width
        target_height_px = int(target_width_px * aspect)
        
        # High-quality resize
        img = img.resize((target_width_px, target_height_px), Image.LANCZOS)
        
        # Save to buffer
        img_buffer = BytesIO()
        img.save(img_buffer, format="JPEG", quality=95, dpi=(dpi, dpi))
        img_buffer.seek(0)
        
        # Add to PDF
        pdf.image(img_buffer, 
                 x=(210 - target_width_mm)/2,  # Center horizontally
                 y=30,
                 w=target_width_mm)
        
        # Add caption
        pdf.set_y(30 + (target_width_mm * aspect * mm_to_inch * 25.4) + 5)
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(108, 117, 125)
        pdf.cell(0, 10, f'Figure {i+1}: Data Visualization', 0, 1, 'C')
    
    return pdf.output(dest="S").encode("UTF-8")

def handle_missing_data(df, strategy):
    if strategy == "Drop Rows":
        return df.dropna()
    elif strategy == "Fill with Mean":
        return df.fillna(df.mean())
    elif strategy == "Fill with Median":
        return df.fillna(df.median())
    return df

def send_data_to_gemini(df, industry, goal, user_query=None):
    df_subset = df.head(100)
    data_json = df_subset.to_json(orient="records", date_format='iso')
    
    prompt = f"""
    You are a business intelligence assistant analyzing {industry} data for {goal}. 
    Generate Python code using Plotly Express wrapped in ```python blocks. Follow these steps:

    1. Identify major KPIs from this data:
    {data_json}
    
    2. For each KPI:
    - Create a meaningful visualization using Plotly Express
    - Use st.plotly_chart(fig) to display it
    - Add clear titles and axis labels
    - Use Streamlit layout components
    
    3. Provide a brief insight after each chart

    Format response EXACTLY like:
    ```python
    # KPI 1 Visualization
    fig1 = px.line(df, x='...', y='...')
    st.plotly_chart(fig1)
    ```
    Insight: [Your insight here]

    ```python
    # KPI 2 Visualization 
    fig2 = px.bar(df, x='...', y='...')
    st.plotly_chart(fig2)
    ```
    Insight: [Your insight here]

    4. Include a final summary with SWOT analysis and recommendations.
    """
    if user_query:
        prompt += f"\nUser Query: {user_query}"
    
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

def main():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Business Intelligence Assistant</h1>", unsafe_allow_html=True)
    st.warning("⚠️ This app executes AI-generated code. Only use with trusted data sources.")

    uploaded_file = st.file_uploader("Upload business data (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df.empty:
            st.error("Uploaded file is empty. Please upload a valid dataset.")
            return
        
        if df.isnull().sum().any():
            st.warning("⚠️ Missing values detected in the dataset.")
            strategy = st.selectbox("Handle missing values by:", ["None", "Drop Rows", "Fill with Mean", "Fill with Median"])
            if strategy != "None":
                df = handle_missing_data(df, strategy)
        
        st.sidebar.header("Data Customization")
        columns = st.sidebar.multiselect("Select columns for analysis", df.columns.tolist(), default=df.columns.tolist())
        df_filtered = df[columns]

        if 'date' in df_filtered.columns:
            min_date = df_filtered['date'].min()
            max_date = df_filtered['date'].max()
            date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
            df_filtered = df_filtered[(df_filtered['date'] >= date_range[0]) & (df_filtered['date'] <= date_range[1])]

        if len(df_filtered) > 1000:
            sample_size = st.sidebar.slider("Sample size for analysis", 100, len(df_filtered), 1000)
            df_filtered = df_filtered.sample(sample_size)

        if st.checkbox("View Data"):
            st.write("### Processed Data Preview")
            st.dataframe(df_filtered)

        st.sidebar.header("Business Context")
        industry = st.sidebar.selectbox("Industry Sector", ["Retail", "E-commerce", "Manufacturing", "Healthcare", "Finance"])
        goal = st.sidebar.selectbox("Primary Objective", ["Revenue Growth", "Cost Optimization", "Customer Experience", "Operational Efficiency"])

        user_query = st.text_input("Ask a question about your data (e.g., 'Show me sales trends over time')")

        if st.button("Analyze Data & Generate Report"):
            with st.spinner("Analyzing data with AI..."):
                report = send_data_to_gemini(df_filtered, industry, goal, user_query)
            try:
                code_blocks = re.findall(r'```python\n(.*?)\n```', report, re.DOTALL)
                insights = re.findall(r'Insight:\s*(.*?)(?=\n\s*```|$)', report, re.DOTALL)

                if not code_blocks:
                    st.error("No code found in response.")
                    return

                exec_env = {'df': df_filtered, 'st': st, 'px': px, 'pd': pd, 'np': np, 'datetime': datetime}
                images = []

                for i, code in enumerate(code_blocks):
                    with st.container():
                        try:
                            exec(code, exec_env)
                            
                            # Retrieve figure with proper naming convention
                            fig_name = f'fig{i+1}'
                            fig = exec_env.get(fig_name)
                            
                            if fig and hasattr(fig, 'data') and len(fig.data) > 0:
                                # Generate high-res image
                                img_bytes = pio.to_image(fig, format="png", scale=3)
                                images.append(BytesIO(img_bytes))
                                
                            if i < len(insights):
                                st.success(f"**Insight {i+1}:** {insights[i].strip()}")
                        except Exception as e:
                            st.error(f"Error executing KPI {i+1}: {str(e)}")

                # Generate PDF report
                report_content = f"Industry: {industry}\nObjective: {goal}\n\nKey Insights:\n" + "\n".join([f"{i+1}. {insight}" for i, insight in enumerate(insights)])
                pdf_report = create_pdf_report(report_content, images)
                st.download_button("Download Full Report", pdf_report, file_name="business_report.pdf", mime="application/pdf")

            except Exception as e:
                st.error(f"Processing Error: {str(e)}")

        # User feedback
        st.sidebar.header("Feedback")
        feedback = st.sidebar.text_area("Provide feedback to help us improve!")
        if st.sidebar.button("Submit Feedback"):
            st.sidebar.success("Thank you for your feedback!")
            
        # User guide
        with st.expander("📖 User Guide"):
            st.write("""
            1. Upload your dataset (CSV or Excel)
            2. Select industry sector and primary objective
            3. Customize data using sidebar filters
            4. Enter specific questions (optional)
            5. Click 'Analyze Data & Generate Report'
            6. View interactive visualizations
            7. Download PDF report
            """)

if __name__ == "__main__":
    main()
