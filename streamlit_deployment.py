import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SmartPremium - Insurance Cost Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    .upload-box {
        background: rgba(30, 41, 59, 0.8);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        margin: 1rem 0;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 12px 40px rgba(245, 87, 108, 0.4);
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.7);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    h1 {
        color: #ffd700;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.9), 0 0 40px rgba(255, 215, 0, 0.8);
        font-weight: 900;
        font-size: 3rem;
        letter-spacing: 2px;
    }
    h2, h3 {
        color: #fcd34d;
    }
    .info-card {
        background: rgba(30, 41, 59, 0.9);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border-left: 4px solid #667eea;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .business-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid rgba(102, 126, 234, 0.4);
        box-shadow: 0 6px 25px rgba(0,0,0,0.4);
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #fbbf24;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(102, 126, 234, 0.3);
        color: #fcd34d;
    }
    p, li {
        color: #e2e8f0;
    }
    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }
    .upload-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.4) 0%, rgba(5, 150, 105, 0.5) 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
        border: 2px solid rgba(16, 185, 129, 0.6);
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.8);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .stMetric label {
        color: #94a3b8 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #fbbf24 !important;
    }
    .stDataFrame {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
    }
    .stExpander {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
    }
    div[data-testid="stExpander"] details summary p {
        color: #fcd34d !important;
        font-weight: bold;
    }
    .uploadedFile {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    input, textarea, select {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    .stSelectbox > div > div {
        background-color: rgba(30, 41, 59, 0.8);
        color: #e2e8f0;
    }
    [data-baseweb="select"] {
        background-color: rgba(30, 41, 59, 0.8);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Helper functions
# -------------------------
def safe_mode_fill(series):
    """Fill missing categorical values with mode."""
    try:
        return series.fillna(series.mode()[0])
    except Exception:
        return series.fillna("missing")

def safe_label_transform(col_series, le):
    """Safely apply saved LabelEncoder to a column, handling unseen labels."""
    s = col_series.astype(str).copy()
    known_classes = set(le.classes_.astype(str))
    fallback = le.classes_[0]
    s = s.apply(lambda x: x if x in known_classes else fallback)
    return le.transform(s)

def preprocess_data(df, label_encoders, feature_cols):
    """Preprocess the input data following the training pipeline."""
    
    # Copy ID if exists
    id_exists = "id" in df.columns
    id_copy = None
    if id_exists:
        id_copy = df["id"].copy()
        df = df.drop(columns=["id"])
    
    # Apply Premium-to-Income filter if both columns exist
    if "Premium Amount" in df.columns and "Annual Income" in df.columns:
        df["Premium_to_Income_Ratio"] = (df["Premium Amount"] / df["Annual Income"]) * 100
        df = df[df["Premium_to_Income_Ratio"] <= 50].copy()
        df = df.drop(columns=["Premium_to_Income_Ratio"])
    
    # Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        df[c] = safe_mode_fill(df[c])
    
    # Convert Policy Start Date to Year, Month, Day
    if "Policy Start Date" in df.columns:
        df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], errors="coerce")
        df["Policy_Year"] = df["Policy Start Date"].dt.year.fillna(0).astype(int)
        df["Policy_Month"] = df["Policy Start Date"].dt.month.fillna(0).astype(int)
        df["Policy_Day"] = df["Policy Start Date"].dt.day.fillna(0).astype(int)
        df = df.drop(columns=["Policy Start Date"])
    else:
        df["Policy_Year"] = 0
        df["Policy_Month"] = 0
        df["Policy_Day"] = 0
    
    # Apply log(1+x) transform
    for col in ["Annual Income", "Previous Claims"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    
    # Apply saved LabelEncoders
    if label_encoders:
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = safe_label_transform(df[col], le)
    
    # Reindex to match model feature order
    X = df.reindex(columns=feature_cols, fill_value=0)
    
    return X, id_copy

def safe_mean(arr):
    """Return mean if arr has elements, else np.nan."""
    arr = np.asarray(arr)
    return arr.mean() if arr.size > 0 else np.nan

def safe_format_currency(x):
    if pd.isna(x):
        return "N/A"
    return f"${x:,.2f}"

# -------------------------
# Initialize session state
# -------------------------
if 'business_data' not in st.session_state:
    st.session_state.business_data = None
if 'business_predictions' not in st.session_state:
    st.session_state.business_predictions = None

# -------------------------
# Title and sidebar (models)
# -------------------------
st.markdown(
    """
    <h1 style="
        text-align:center;
        color:#FFD700;
        text-shadow: 3px 3px 12px rgba(0,0,0,0.85);
        font-weight:900;
        font-size:3rem;
        padding:0.4rem 0;
    ">
        SmartPremium: Predicting Insurance Costs with Machine Learning
    </h1>
    """,
    unsafe_allow_html=True
)


st.sidebar.header("📁 Upload Model Files")
st.sidebar.markdown("<p style='color: #fcd34d;'><b>Upload all 3 pickle files at once:</b></p>", unsafe_allow_html=True)

uploaded_model_files = st.sidebar.file_uploader(
    "Select all model files (*.pkl)", 
    type=['pkl'], 
    accept_multiple_files=True,
    help="Upload all 3 files: label_encoders.pkl, randomforest_model.pkl, regression_feature_cols.pkl"
)

# Load model files
models_loaded = False
label_encoders = None
rf_model = None
feature_cols = None

if uploaded_model_files:
    try:
        loaded_files = {}
        
        for file in uploaded_model_files:
            file_name = file.name.lower()
            loaded_files[file_name] = joblib.load(file)
        
        if 'label_encoders.pkl' in loaded_files:
            label_encoders = loaded_files['label_encoders.pkl']
        if 'randomforest_model.pkl' in loaded_files:
            rf_model = loaded_files['randomforest_model.pkl']
        if 'regression_feature_cols.pkl' in loaded_files:
            feature_cols = loaded_files['regression_feature_cols.pkl']
        
        if label_encoders is not None and rf_model is not None and feature_cols is not None:
            models_loaded = True
            st.sidebar.markdown("""
                <div class='upload-success'>
                    <h4 style='color: #ffffff; margin: 0; font-size: 1.1rem; text-shadow: 1px 1px 3px rgba(0,0,0,0.5);'>✅ All Models Loaded!</h4>
                    <p style='color: #ffffff; margin: 0.5rem 0 0 0; font-size: 0.95rem; font-weight: 600;'>
                    ✓ Label Encoders<br>
                    ✓ Random Forest Model<br>
                    ✓ Feature Columns
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            missing = []
            if label_encoders is None:
                missing.append("label_encoders.pkl")
            if rf_model is None:
                missing.append("randomforest_model.pkl")
            if feature_cols is None:
                missing.append("regression_feature_cols.pkl")
            
            st.sidebar.warning(f"⚠️ Missing files: {', '.join(missing)}")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")

# -------------------------
# Main content
# -------------------------
if not models_loaded:
    st.markdown("""
        <div class='info-card'>
            <h3 style='color: #fcd34d;'>👋 Welcome to SmartPremium!</h3>
            <p style='color: #e2e8f0;'>To get started, please upload all required pickle files in the sidebar (you can select all 3 at once):</p>
            <ul style='color: #e2e8f0;'>
                <li><strong style='color: #fbbf24;'>label_encoders.pkl</strong> - Encoding transformations</li>
                <li><strong style='color: #fbbf24;'>randomforest_model.pkl</strong> - Trained prediction model</li>
                <li><strong style='color: #fbbf24;'>regression_feature_cols.pkl</strong> - Feature configuration</li>
            </ul>
            <p style='color: #fcd34d; font-weight: bold; margin-top: 1rem;'>💡 Tip: Hold Ctrl (or Cmd on Mac) to select multiple files at once!</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Batch Prediction", "👤 Single Prediction", "💼 Business Use Cases"])
    
    with tab1:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.subheader("Upload CSV for Batch Prediction")
        st.write("Upload a CSV file with customer data to get premium predictions for multiple records.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Dataset shape: {df.shape}")
                
                with st.expander("📋 Preview uploaded data"):
                    st.dataframe(df.head(10))
                
                if st.button("🚀 Generate Predictions", key="batch_predict"):
                    with st.spinner("Processing data and generating predictions..."):
                        X_test, id_copy = preprocess_data(df.copy(), label_encoders, feature_cols)
                        predictions = rf_model.predict(X_test)
                        predictions = np.asarray(predictions)
                        
                        submission = pd.DataFrame({
                            "id": id_copy if id_copy is not None else range(len(predictions)),
                            "Premium Amount": np.round(predictions, 6)
                        })
                        
                        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                        st.write("✨ Predictions Generated Successfully!")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                        with col2:
                            st.metric("Average Premium", f"${safe_mean(predictions):,.2f}" if len(predictions)>0 else "N/A")
                        
                        st.subheader("📊 Prediction Results")
                        st.dataframe(submission)
                        
                        csv = submission.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        with st.expander("📈 Prediction Statistics"):
                            if len(predictions) > 0:
                                st.write(f"**Minimum Premium:** ${predictions.min():,.2f}")
                                st.write(f"**Maximum Premium:** ${predictions.max():,.2f}")
                                st.write(f"**Median Premium:** ${np.median(predictions):,.2f}")
                                st.write(f"**Standard Deviation:** ${np.std(predictions):,.2f}")
                            else:
                                st.write("No predictions generated.")
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with tab2:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.subheader("Enter Customer Details")
        st.write("Fill in the customer information below to get an insurance premium prediction.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
            education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
        
        with col2:
            occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
            health_score = st.slider("Health Score", min_value=0, max_value=100, value=75)
            location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
            policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
            previous_claims = st.number_input("Previous Claims", min_value=0, max_value=50, value=0)
            vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
        
        with col3:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            insurance_duration = st.number_input("Insurance Duration (years)", min_value=1, max_value=30, value=1)
            policy_start_date = st.date_input("Policy Start Date", value=datetime.now())
            smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])
            exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
            property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])
        
        if st.button("🎯 Predict Premium", key="single_predict"):
            try:
                input_data = pd.DataFrame({
                    "Age": [age],
                    "Gender": [gender],
                    "Annual Income": [annual_income],
                    "Marital Status": [marital_status],
                    "Number of Dependents": [num_dependents],
                    "Education Level": [education],
                    "Occupation": [occupation],
                    "Health Score": [health_score],
                    "Location": [location],
                    "Policy Type": [policy_type],
                    "Previous Claims": [previous_claims],
                    "Vehicle Age": [vehicle_age],
                    "Credit Score": [credit_score],
                    "Insurance Duration": [insurance_duration],
                    "Policy Start Date": [policy_start_date.strftime("%Y-%m-%d")],
                    "Smoking Status": [smoking_status],
                    "Exercise Frequency": [exercise_frequency],
                    "Property Type": [property_type]
                })
                
                with st.spinner("Calculating premium..."):
                    X_test, _ = preprocess_data(input_data, label_encoders, feature_cols)
                    prediction = rf_model.predict(X_test)[0]
                    
                    st.markdown(f"""
                        <div class='prediction-box'>
                            <h2>Predicted Insurance Premium</h2>
                            <h1>${prediction:,.2f}</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if prediction < 1000:
                        risk_level = "Low Risk"
                        risk_color = "green"
                    elif prediction < 3000:
                        risk_level = "Medium Risk"
                        risk_color = "orange"
                    else:
                        risk_level = "High Risk"
                        risk_color = "red"
                    
                    st.markdown(f"""
                        <div class='info-card'>
                            <h3 style='color: #fcd34d;'>Risk Assessment: <span style='color: {risk_color}'>{risk_level}</span></h3>
                            <p style='color: #e2e8f0;'>This premium is calculated based on the customer's profile and policy details.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error generating prediction: {str(e)}")
    
    with tab3:
        st.markdown("""
            <div class='info-card'>
                <h2 style='color: #ffd700;'>💼 Business Use Cases</h2>
                <p style='color: #e2e8f0;'>Upload your dataset to analyze business insights and applications across different industries.</p>
            </div>
        """, unsafe_allow_html=True)
        
        business_file = st.file_uploader("Upload CSV for Business Analysis", type=['csv'], key="business_upload")
        
        if business_file is not None:
            try:
                business_df = pd.read_csv(business_file)
                st.success(f"✅ Dataset loaded! Total records: {len(business_df)}")
                
                with st.expander("📋 Preview Dataset"):
                    st.dataframe(business_df.head(10))
                
                if st.button("🔍 Analyze Business Use Cases", key="analyze_business"):
                    with st.spinner("Generating predictions and analyzing data..."):
                        X_test, id_copy = preprocess_data(business_df.copy(), label_encoders, feature_cols)
                        predictions = rf_model.predict(X_test)
                        predictions = np.asarray(predictions)
                        
                        st.session_state.business_data = business_df.copy()
                        st.session_state.business_predictions = predictions
                        
                        st.success("✅ Analysis complete! Explore the use cases below.")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        # Display business use cases if data is available
        if st.session_state.business_data is not None and st.session_state.business_predictions is not None:
            df_business = st.session_state.business_data
            preds = np.asarray(st.session_state.business_predictions)
            
            # Precompute health_summary and df_health so both Use Case 2 and 3 can use them
            df_health = None
            health_summary = None
            if "Health Score" in df_business.columns:
                try:
                    df_health = pd.DataFrame({
                        'Health Score': df_business['Health Score'].astype(float),
                        'Predicted Premium': preds
                    })
                    df_health['Health Category'] = pd.cut(
                        df_health['Health Score'],
                        bins=[0, 50, 70, 85, 100],
                        labels=['Poor', 'Fair', 'Good', 'Excellent'],
                        include_lowest=True
                    )
                    health_summary = df_health.groupby('Health Category')['Predicted Premium'].agg(['mean', 'count'])
                except Exception:
                    df_health = None
                    health_summary = None
            
            st.markdown("---")
            
            # Use Case 1: Insurance Companies
            with st.expander("💰 Insurance Companies - Premium Pricing Optimization", expanded=False):
                st.markdown("""
                    <div class='business-card'>
                        <h3 style='color: #ffd700;'>Premium Pricing Strategy</h3>
                        <p style='color: #e2e8f0;'>Optimize premium pricing based on risk factors to maximize profitability while remaining competitive.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                low_risk = preds < 1000
                medium_risk = (preds >= 1000) & (preds < 3000)
                high_risk = preds >= 3000
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Low Risk Customers", f"{low_risk.sum()} ({low_risk.sum()/len(preds)*100:.1f}%)" if len(preds)>0 else "0 (0.0%)")
                    if low_risk.sum() > 0:
                        st.write(f"Avg Premium: {safe_format_currency(safe_mean(preds[low_risk]))}")
                with col2:
                    st.metric("Medium Risk Customers", f"{medium_risk.sum()} ({medium_risk.sum()/len(preds)*100:.1f}%)" if len(preds)>0 else "0 (0.0%)")
                    if medium_risk.sum() > 0:
                        st.write(f"Avg Premium: {safe_format_currency(safe_mean(preds[medium_risk]))}")
                with col3:
                    st.metric("High Risk Customers", f"{high_risk.sum()} ({high_risk.sum()/len(preds)*100:.1f}%)" if len(preds)>0 else "0 (0.0%)")
                    if high_risk.sum() > 0:
                        st.write(f"Avg Premium: {safe_format_currency(safe_mean(preds[high_risk]))}")
                
                st.markdown("#### 🎯 Key Insights")
                st.write(f"- **Total Revenue Potential:** {safe_format_currency(preds.sum() if len(preds)>0 else np.nan)}")
                st.write(f"- **Average Premium:** {safe_format_currency(safe_mean(preds))}")
                st.write(f"- **Premium Range:** {safe_format_currency(preds.min() if len(preds)>0 else np.nan)} - {safe_format_currency(preds.max() if len(preds)>0 else np.nan)}")
                
                if "Policy Type" in df_business.columns:
                    st.markdown("#### 📊 Premium by Policy Type")
                    policy_analysis = pd.DataFrame({
                        'Policy Type': df_business['Policy Type'],
                        'Premium': preds
                    }).groupby('Policy Type')['Premium'].agg(['mean', 'count', 'sum'])
                    # display without .style if issues; formatting applied to the numeric columns
                    st.dataframe(policy_analysis.rename(columns={'mean':'Average','count':'Count','sum':'Total'}).style.format({'Average': '${:,.2f}', 'Total': '${:,.2f}'}))
            
            # Use Case 2: Financial Institutions
            with st.expander("📊 Financial Institutions - Risk Assessment for Loans", expanded=False):
                st.markdown("""
                    <div class='business-card'>
                        <h3 style='color: #ffd700;'>Loan Risk Assessment</h3>
                        <p style='color: #e2e8f0;'>Assess customer risk for loan approvals tied to insurance policies.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if "Credit Score" in df_business.columns:
                    df_credit = pd.DataFrame({
                        'Credit Score': df_business['Credit Score'].astype(float),
                        'Predicted Premium': preds
                    })
                    
                    df_credit['Credit Category'] = pd.cut(df_credit['Credit Score'], 
                                                           bins=[0, 580, 670, 740, 850],
                                                           labels=['Poor', 'Fair', 'Good', 'Excellent'],
                                                           include_lowest=True)
                    
                    credit_summary = df_credit.groupby('Credit Category')['Predicted Premium'].agg(['mean', 'count'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 🏥 Health Score Impact")
                        if health_summary is not None:
                            st.dataframe(health_summary.style.format({'mean': '${:,.2f}'}))
                        else:
                            st.write("No Health Score column available in uploaded dataset.")
                    
                    with col2:
                        st.markdown("#### 💊 Cost Estimation Insights")
                        if df_health is not None:
                            poor_health = (df_health['Health Category'] == 'Poor').sum()
                            fair_health = (df_health['Health Category'] == 'Fair').sum()
                            good_health = (df_health['Health Category'] == 'Good').sum()
                            excellent_health = (df_health['Health Category'] == 'Excellent').sum()
                            
                            st.write(f"🔴 **High Cost Risk (Poor):** {poor_health} patients")
                            st.write(f"🟡 **Moderate Cost (Fair):** {fair_health} patients")
                            st.write(f"🟢 **Low Cost (Good):** {good_health} patients")
                            st.write(f"✅ **Minimal Cost (Excellent):** {excellent_health} patients")
                        else:
                            st.write("Health distribution not available.")
                    
                    st.markdown("#### 💰 Healthcare Budget Planning")
                    avg_premium = safe_mean(preds)
                    total_cost = preds.sum() if len(preds) > 0 else 0
                    st.write(f"- **Average Expected Healthcare Cost per Patient:** {safe_format_currency(avg_premium)}")
                    st.write(f"- **Total Budget Required:** {safe_format_currency(total_cost)}")
                    st.write(f"- **Cost Range:** {safe_format_currency(preds.min() if len(preds)>0 else np.nan)} - {safe_format_currency(preds.max() if len(preds)>0 else np.nan)}")
                    
                    if "Age" in df_business.columns:
                        st.markdown("#### 👥 Age-Based Cost Analysis")
                        df_age_health = pd.DataFrame({
                            'Age': df_business['Age'].astype(float),
                            'Premium': preds
                        })
                        df_age_health['Age Group'] = pd.cut(df_age_health['Age'], 
                                                             bins=[0, 30, 45, 60, 100],
                                                             labels=['18-30', '31-45', '46-60', '60+'],
                                                             include_lowest=True)
                        age_summary = df_age_health.groupby('Age Group')['Premium'].agg(['mean', 'count'])
                        st.dataframe(age_summary.style.format({'mean': '${:,.2f}'}))
                else:
                    st.write("Credit Score column not present in dataset; skip loan risk analysis.")
            
            # Use Case 3: Healthcare Providers
            with st.expander("🧑‍⚕️ Healthcare Providers - Future Cost Estimation", expanded=False):
                st.markdown("""
                    <div class='business-card'>
                        <h3 style='color: #ffd700;'>Healthcare Cost Forecasting</h3>
                        <p style='color: #e2e8f0;'>Estimate future healthcare costs for patients based on their insurance profiles.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if df_health is not None:
                    st.markdown("#### Health Category Summary")
                    st.dataframe(health_summary.style.format({'mean': '${:,.2f}'}))
                    
                    st.markdown("#### Health Category Counts")
                    st.write(df_health['Health Category'].value_counts().to_frame(name='count'))
                    
                    st.markdown("#### Additional Insights")
                    st.write(f"- Average premium for Poor health: {safe_format_currency(safe_mean(df_health.loc[df_health['Health Category']=='Poor','Predicted Premium'].values))}")
                    st.write(f"- Average premium for Excellent health: {safe_format_currency(safe_mean(df_health.loc[df_health['Health Category']=='Excellent','Predicted Premium'].values))}")
                else:
                    st.write("No Health Score column found in the dataset.")
            
            # Use Case 4: Customer Service Optimization
            with st.expander("🔍 Customer Service Optimization - Real-Time Insurance Quotes", expanded=False):
                st.markdown("""
                    <div class='business-card'>
                        <h3 style='color: #ffd700;'>Instant Quote Generation</h3>
                        <p style='color: #e2e8f0;'>Provide real-time, data-driven insurance quotes to customers instantly.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 🚀 Quick Quote Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Quotes", len(preds))
                with col2:
                    st.metric("Avg Quote Time", "< 1 sec")
                with col3:
                    st.metric("Quote Range", f"{safe_format_currency(preds.min() if len(preds)>0 else np.nan)} - {safe_format_currency(preds.max() if len(preds)>0 else np.nan)}")
                with col4:
                    st.metric("Median Quote", f"{safe_format_currency(np.median(preds) if len(preds)>0 else np.nan)}")
                
                st.markdown("#### 📊 Quote Distribution by Risk Level")
                low_quotes = (preds < 1000).sum()
                med_quotes = ((preds >= 1000) & (preds < 3000)).sum()
                high_quotes = (preds >= 3000).sum()
                
                quote_dist = pd.DataFrame({
                    'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
                    'Number of Quotes': [low_quotes, med_quotes, high_quotes],
                    'Percentage': [
                        f"{(low_quotes/len(preds)*100):.1f}%" if len(preds)>0 else "0.0%",
                        f"{(med_quotes/len(preds)*100):.1f}%" if len(preds)>0 else "0.0%",
                        f"{(high_quotes/len(preds)*100):.1f}%" if len(preds)>0 else "0.0%"
                    ],
                    'Avg Premium': [
                        f"{safe_format_currency(safe_mean(preds[preds < 1000]))}" if low_quotes > 0 else "N/A",
                        f"{safe_format_currency(safe_mean(preds[(preds >= 1000) & (preds < 3000)]))}" if med_quotes > 0 else "N/A",
                        f"{safe_format_currency(safe_mean(preds[preds >= 3000]))}" if high_quotes > 0 else "N/A"
                    ]
                })
                st.dataframe(quote_dist)
                
                st.markdown("#### 💡 Customer Service Benefits")
                st.write("- **Instant Quotes:** Generate accurate quotes in under 1 second")
                st.write("- **Consistency:** All quotes follow the same data-driven methodology")
                st.write("- **Transparency:** Clear risk-based pricing for customers")
                st.write("- **Efficiency:** Reduce quote generation time by 95%")
                st.write("- **Scalability:** Handle unlimited quote requests simultaneously")
                
                if "Policy Type" in df_business.columns:
                    st.markdown("#### 📋 Quotes by Policy Type")
                    policy_quotes = pd.DataFrame({
                        'Policy Type': df_business['Policy Type'],
                        'Premium': preds
                    }).groupby('Policy Type').agg(
                        Count=('Premium', 'count'),
                        Average=('Premium', 'mean'),
                        Minimum=('Premium', 'min'),
                        Maximum=('Premium', 'max')
                    )
                    st.dataframe(policy_quotes.style.format({
                        'Average': '${:,.2f}',
                        'Minimum': '${:,.2f}',
                        'Maximum': '${:,.2f}'
                    }))
                
                st.markdown("#### 🎯 Customer Segmentation for Targeted Service")
                if "Annual Income" in df_business.columns:
                    df_income = pd.DataFrame({
                        'Annual Income': df_business['Annual Income'],
                        'Premium': preds
                    })
                    
                    st.write("Understanding customer segments helps tailor service approaches:")
                    st.write("- **High-income, low premium:** Potential for upselling premium policies")
                    st.write("- **Low-income, high premium:** May need payment plan options")
                    st.write("- **Medium-income, medium premium:** Standard service approach")
