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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .upload-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
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
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .info-card {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
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
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = safe_label_transform(df[col], le)
    
    # Reindex to match model feature order
    X = df.reindex(columns=feature_cols, fill_value=0)
    
    return X, id_copy if id_exists else None

# Title and header
st.markdown("<h1 style='text-align: center;'>💰 SmartPremium Insurance Cost Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'>Predict insurance premiums using advanced machine learning</p>", unsafe_allow_html=True)

# Sidebar for model uploads
st.sidebar.header("📁 Upload Model Files")
st.sidebar.markdown("Upload all required pickle files to get started:")

label_encoders_file = st.sidebar.file_uploader("Label Encoders (label_encoders.pkl)", type=['pkl'])
model_file = st.sidebar.file_uploader("Random Forest Model (randomforest_model.pkl)", type=['pkl'])
feature_cols_file = st.sidebar.file_uploader("Feature Columns (regression_feature_cols.pkl)", type=['pkl'])

# Load model files
models_loaded = False
if label_encoders_file and model_file and feature_cols_file:
    try:
        label_encoders = joblib.load(label_encoders_file)
        rf_model = joblib.load(model_file)
        feature_cols = joblib.load(feature_cols_file)
        models_loaded = True
        st.sidebar.success("✅ All models loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")

# Main content
if not models_loaded:
    st.markdown("""
        <div class='info-card'>
            <h3>👋 Welcome to SmartPremium!</h3>
            <p>To get started, please upload all required pickle files in the sidebar:</p>
            <ul>
                <li><strong>label_encoders.pkl</strong> - Encoding transformations</li>
                <li><strong>randomforest_model.pkl</strong> - Trained prediction model</li>
                <li><strong>regression_feature_cols.pkl</strong> - Feature configuration</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
else:
    # Create tabs for different prediction modes
    tab1, tab2 = st.tabs(["📊 Batch Prediction", "👤 Single Prediction"])
    
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
                            st.metric("Average Premium", f"${np.mean(predictions):,.2f}")
                        
                        st.subheader("📊 Prediction Results")
                        st.dataframe(submission)
                        
                        # Download button
                        csv = submission.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Statistics
                        with st.expander("📈 Prediction Statistics"):
                            st.write(f"**Minimum Premium:** ${predictions.min():,.2f}")
                            st.write(f"**Maximum Premium:** ${predictions.max():,.2f}")
                            st.write(f"**Median Premium:** ${np.median(predictions):,.2f}")
                            st.write(f"**Standard Deviation:** ${np.std(predictions):,.2f}")
                        
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
                # Create dataframe from inputs
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
                    
                    # Risk assessment
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
                            <h3>Risk Assessment: <span style='color: {risk_color}'>{risk_level}</span></h3>
                            <p>This premium is calculated based on the customer's profile and policy details.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error generating prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 1rem;'>
        <p>💡 SmartPremium - Powered by Machine Learning | Built with Streamlit</p>
        <p style='font-size: 0.9rem;'>Insurance premium predictions based on customer characteristics and policy details</p>
    </div>
""", unsafe_allow_html=True)