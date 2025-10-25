import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="SmartPremium - Insurance Cost Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin: 20px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px 0;
    }
    .info-text {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_models():
    """Load all saved models and encoders"""
    try:
        label_encoders = joblib.load(r"D:\prakash\Smart_Premium_New\playground-series-s4e12 (3)\label_encoders.pkl")
        rf_model = joblib.load(r"D:\prakash\Smart_Premium_New\playground-series-s4e12 (3)\randomforest_model.pkl")
        feature_cols = joblib.load(r"D:\prakash\Smart_Premium_New\playground-series-s4e12 (3)\regression_feature_cols.pkl")
        return label_encoders, rf_model, feature_cols
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def safe_label_transform(value, le, col_name):
    """Safely apply saved LabelEncoder"""
    value_str = str(value)
    known_classes = set(le.classes_.astype(str))
    if value_str not in known_classes:
        value_str = le.classes_[0]
    return le.transform([value_str])[0]

def preprocess_input(input_data, label_encoders):
    """Preprocess user input to match training pipeline"""
    df = pd.DataFrame([input_data])
    
    # Convert Policy Start Date to Year, Month, Day
    if "Policy Start Date" in df.columns:
        df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"])
        df["Policy_Year"] = df["Policy Start Date"].dt.year
        df["Policy_Month"] = df["Policy Start Date"].dt.month
        df["Policy_Day"] = df["Policy Start Date"].dt.day
        df = df.drop(columns=["Policy Start Date"])
    
    # Apply log(1+x) transformation
    for col in ["Annual Income", "Previous Claims"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    
    # Apply label encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = safe_label_transform(df[col].iloc[0], le, col)
    
    return df

# Main App
def main():
    # Header
    st.markdown("<h1>💰 SmartPremium Insurance Cost Predictor</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-text'>
        <b>Welcome!</b> This AI-powered tool predicts insurance premiums based on customer characteristics 
        and policy details. Fill in the information below to get an instant quote.
        </div>
        """, unsafe_allow_html=True)
    
    # Load models
    label_encoders, rf_model, feature_cols = load_models()
    
    if label_encoders is None or rf_model is None:
        st.error("⚠️ Models not found. Please ensure all model files are in the correct directory.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/insurance.png", width=150)
        st.markdown("### 📋 About")
        st.info("""
        This application uses machine learning to predict insurance premiums based on:
        - Personal Information
        - Financial Details
        - Health Metrics
        - Policy Information
        - Lifestyle Factors
        """)
        st.markdown("### 📊 Model Info")
        st.success("✅ Random Forest Model\n\n✅ High Accuracy\n\n✅ Real-time Predictions")
    
    # Main form
    with st.form("prediction_form"):
        st.markdown("### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Your current age")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        
        with col2:
            education = st.selectbox("Education Level", 
                                    ["High School", "Bachelor's", "Master's", "PhD"])
            occupation = st.selectbox("Occupation", 
                                     ["Employed", "Self-Employed", "Unemployed"])
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        
        with col3:
            location = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"])
            property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])
        
        st.markdown("---")
        st.markdown("### 💵 Financial Information")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            annual_income = st.number_input("Annual Income ($)", 
                                           min_value=10000, max_value=1000000, 
                                           value=50000, step=5000,
                                           help="Your yearly income in dollars")
        
        with col5:
            credit_score = st.number_input("Credit Score", 
                                          min_value=300, max_value=850, 
                                          value=700,
                                          help="Your credit score (300-850)")
        
        with col6:
            vehicle_age = st.number_input("Vehicle Age (years)", 
                                         min_value=0, max_value=30, 
                                         value=5)
        
        st.markdown("---")
        st.markdown("### 🏥 Health & Lifestyle")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            health_score = st.slider("Health Score", 
                                    min_value=0, max_value=100, 
                                    value=75,
                                    help="Self-assessed health score (0-100)")
            smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])
        
        with col8:
            exercise_freq = st.selectbox("Exercise Frequency", 
                                        ["Daily", "Weekly", "Monthly", "Rarely"])
        
        with col9:
            previous_claims = st.number_input("Previous Claims", 
                                             min_value=0, max_value=50, 
                                             value=0,
                                             help="Number of previous insurance claims")
        
        st.markdown("---")
        st.markdown("### 📄 Policy Information")
        col10, col11, col12 = st.columns(3)
        
        with col10:
            policy_type = st.selectbox("Policy Type", 
                                      ["Basic", "Comprehensive", "Premium"])
        
        with col11:
            insurance_duration = st.number_input("Insurance Duration (years)", 
                                                min_value=1, max_value=30, 
                                                value=1)
        
        with col12:
            policy_start_date = st.date_input("Policy Start Date", 
                                             value=datetime.now())
        
        # Submit button
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Premium")
    
    # Prediction
    if submitted:
        with st.spinner("🔄 Calculating your premium..."):
            # Prepare input data
            input_data = {
                "Age": age,
                "Gender": gender,
                "Annual Income": annual_income,
                "Marital Status": marital_status,
                "Number of Dependents": dependents,
                "Education Level": education,
                "Occupation": occupation,
                "Health Score": health_score,
                "Location": location,
                "Policy Type": policy_type,
                "Previous Claims": previous_claims,
                "Vehicle Age": vehicle_age,
                "Credit Score": credit_score,
                "Insurance Duration": insurance_duration,
                "Policy Start Date": policy_start_date,
                "Smoking Status": smoking_status,
                "Exercise Frequency": exercise_freq,
                "Property Type": property_type
            }
            
            try:
                # Preprocess and predict
                processed_data = preprocess_input(input_data, label_encoders)
                X_test = processed_data.reindex(columns=feature_cols, fill_value=0)
                prediction = rf_model.predict(X_test)[0]
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
                
                with col_res2:
                    st.markdown(f"""
                        <div class='prediction-box'>
                            <h2>Your Estimated Premium</h2>
                            <h1 style='font-size: 48px; margin: 20px 0;'>${prediction:,.2f}</h1>
                            <p style='font-size: 18px;'>per year</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 📊 Premium Breakdown")
                col_insight1, col_insight2, col_insight3 = st.columns(3)
                
                with col_insight1:
                    monthly = prediction / 12
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Monthly Payment</h4>
                            <h2>${monthly:,.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_insight2:
                    ratio = (prediction / annual_income) * 100
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4>% of Annual Income</h4>
                            <h2>{ratio:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_insight3:
                    per_day = prediction / 365
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4>Daily Cost</h4>
                            <h2>${per_day:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualization
                st.markdown("### 💡 Cost Comparison")
                fig = go.Figure(data=[
                    go.Bar(name='Your Premium', x=['Annual', 'Monthly', 'Daily'], 
                          y=[prediction, monthly, per_day],
                          marker_color=['#667eea', '#764ba2', '#f093fb'])
                ])
                fig.update_layout(
                    title="Premium Breakdown Across Time Periods",
                    yaxis_title="Amount ($)",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors
                st.markdown("### ⚠️ Factors Affecting Your Premium")
                factors = []
                
                if smoking_status == "Yes":
                    factors.append("🚬 Smoking increases premium")
                if age > 50:
                    factors.append("👴 Higher age increases premium")
                if previous_claims > 3:
                    factors.append("📋 Multiple previous claims increase premium")
                if health_score < 50:
                    factors.append("🏥 Lower health score increases premium")
                if exercise_freq in ["Rarely", "Monthly"]:
                    factors.append("🏃 Low exercise frequency increases premium")
                
                if factors:
                    for factor in factors:
                        st.warning(factor)
                else:
                    st.success("✅ Great! You have optimal risk factors for a good premium rate.")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check your inputs and try again.")

if __name__ == "__main__":
    main()