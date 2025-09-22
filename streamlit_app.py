
import streamlit as st
import requests
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="CoinPilot Prediction Service",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🚀 CoinPilot Premium User Conversion Prediction")
st.markdown("---")
st.markdown("### Predict whether users will convert to premium using machine learning")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration Settings")

# API configuration
api_url = st.sidebar.text_input(
    "API Address", 
    value="http://localhost:8000",
    help="FastAPI server address"
)

# Mode selection
use_offline = st.sidebar.checkbox(
    "Use Offline Mode", 
    value=False,
    help="If enabled, use the local model instead of calling the API"
)

# Offline mode configuration
offline_model_path = st.sidebar.text_input(
    "Offline Model Path",
    value="coinpilot_prediction_pipeline.joblib",
    help="Path to local model file"
)

# Check API health status
def check_api_health(api_url):
    """Check API health status"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except:
        return False, None

# Load offline model
@st.cache_resource
def load_offline_model(model_path):
    """Load offline model"""
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

# Preprocess input data
def preprocess_input_for_offline(user_input):
    """Preprocess input data for offline mode"""
    # Create country one-hot encoding
    country_columns = ['country_ID', 'country_MY', 'country_PH', 'country_SG', 'country_TH', 'country_VN']
    country_encoded = {col: 1 if col == f'country_{user_input["country"]}' else 0 for col in country_columns}
    
    # Build feature dictionary
    features = {
        'age': user_input['age'],
        'tenure_months': user_input['tenure_months'],
        'income_monthly': user_input['income_monthly'],
        'savings_rate': user_input['savings_rate'],
        'risk_score': user_input['risk_score'],
        'app_opens_7d': user_input['app_opens_7d'],
        'sessions_7d': user_input['sessions_7d'],
        'avg_session_min': user_input['avg_session_min'],
        'alerts_opt_in': user_input['alerts_opt_in'],
        'auto_invest': user_input['auto_invest'],
        'equity_pct': user_input['equity_pct'],
        'bond_pct': user_input['bond_pct'],
        'cash_pct': user_input['cash_pct'],
        'crypto_pct': user_input['crypto_pct'],
        **country_encoded
    }
    
    return pd.DataFrame([features])

# Get confidence level
def get_confidence_level(probability):
    """Determine confidence level based on probability"""
    if probability >= 0.8:
        return "High"
    elif probability >= 0.6:
        return "Medium"
    else:
        return "Low"

# Main interface
def main():
    # Check API status
    if not use_offline:
        with st.spinner("Checking API status..."):
            api_healthy, health_data = check_api_health(api_url)
        
        if api_healthy:
            st.sidebar.success("✅ API connection OK")
            if health_data:
                st.sidebar.json(health_data)
        else:
            st.sidebar.error("❌ API connection failed")
            st.sidebar.info("💡 Suggest switching to offline mode")
    else:
        st.sidebar.info("🔧 Using offline mode")
    
    # User input form
    st.header("📝 User Information Input")
    
    # Two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Basic Information")
        age = st.slider("Age", min_value=18, max_value=100, value=30, help="User age")
        tenure_months = st.slider("Tenure (months)", min_value=1, max_value=60, value=12, help="Months since user started using app")
        income_monthly = st.number_input("Monthly Income", min_value=0.0, value=5000.0, step=100.0, help="User monthly income")
        savings_rate = st.slider("Savings Rate", min_value=0.0, max_value=1.0, value=0.3, step=0.01, help="User savings rate")
        risk_score = st.slider("Risk Score", min_value=0.0, max_value=100.0, value=50.0, step=1.0, help="User risk score")
    
    with col2:
        st.subheader("📱 App Usage")
        app_opens_7d = st.number_input("App Opens in Last 7 Days", min_value=0, value=10, help="How many times the app was opened in the last 7 days")
        sessions_7d = st.number_input("Sessions in Last 7 Days", min_value=0, value=15, help="How many sessions in the last 7 days")
        avg_session_min = st.number_input("Average Session Duration (minutes)", min_value=0.0, value=5.0, step=0.1, help="Average length of each session")
        alerts_opt_in = st.selectbox("Alerts Subscription", [0, 1], index=1, help="Whether the user subscribed to alerts")
        auto_invest = st.selectbox("Auto-Invest Enabled", [0, 1], index=1, help="Whether the user enabled auto-invest")
    
    # Portfolio configuration
    st.subheader("💰 Portfolio Allocation")
    col3, col4 = st.columns(2)
    
    with col3:
        country = st.selectbox("Country", ["SG", "MY", "TH", "PH", "VN", "ID"], index=0, help="User's country")
    
    with col4:
        st.markdown("**Portfolio Percentages** (should sum to 100%)")
        equity_pct = st.slider("Equity %", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
        bond_pct = st.slider("Bond %", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        cash_pct = st.slider("Cash %", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        crypto_pct = st.slider("Crypto %", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    
    # Check portfolio sum
    total_pct = equity_pct + bond_pct + cash_pct + crypto_pct
    if abs(total_pct - 100.0) > 0.1:
        st.warning(f"⚠️ Portfolio total is {total_pct:.1f}%. Please adjust to 100%.")
    
    # Prediction button
    if st.button("🔮 Start Prediction", type="primary"):
        # Prepare input data
        input_data = {
            "age": age,
            "tenure_months": tenure_months,
            "income_monthly": income_monthly,
            "savings_rate": savings_rate,
            "risk_score": risk_score,
            "app_opens_7d": app_opens_7d,
            "sessions_7d": sessions_7d,
            "avg_session_min": avg_session_min,
            "alerts_opt_in": alerts_opt_in,
            "auto_invest": auto_invest,
            "country": country,
            "equity_pct": equity_pct,
            "bond_pct": bond_pct,
            "cash_pct": cash_pct,
            "crypto_pct": crypto_pct
        }
        
        # Execute prediction
        with st.spinner("Making prediction..."):
            try:
                if use_offline:
                    # Offline prediction
                    model, error = load_offline_model(offline_model_path)
                    if model is None:
                        st.error(f"❌ Failed to load offline model: {error}")
                        return
                    
                    processed_data = preprocess_input_for_offline(input_data)
                    prediction = model.predict(processed_data)[0]
                    probability = model.predict_proba(processed_data)[0][1]
                    confidence = get_confidence_level(probability)
                    
                    result = {
                        "prediction": int(prediction),
                        "probability": float(probability),
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat(),
                        "mode": "offline"
                    }
                    
                    st.success("✅ Offline prediction completed")
                    
                else:
                    # Online prediction
                    response = requests.post(f"{api_url}/predict", json=input_data, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        result["mode"] = "online"
                        st.success("✅ API prediction completed")
                    else:
                        st.error(f"❌ API prediction failed: {response.text}")
                        return
                        
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                return
        
        # Show results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            prediction_text = "Will Convert" if result["prediction"] == 1 else "Will Not Convert"
            prediction_icon = "🟢" if result["prediction"] == 1 else "🔴"
            st.metric("Prediction", f"{prediction_icon} {prediction_text}")
        
        with col2:
            st.metric("Conversion Probability", f"{result['probability']:.1%}")
        
        with col3:
            confidence_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[result["confidence"]]
            st.metric("Confidence", f"{confidence_icon} {result['confidence']}")
        
        with col4:
            mode_icon = "🌐" if result["mode"] == "online" else "💻"
            st.metric("Mode", f"{mode_icon} {result['mode']}")
        
        # Detailed results
        st.subheader("📈 Detailed Analysis")
        
        prob = result['probability']
        st.progress(prob)
        st.caption(f"Conversion Probability: {prob:.2%}")
        
        confidence_explanation = {
            "High": "The model is very confident in this prediction.",
            "Medium": "The model is somewhat confident in this prediction.",
            "Low": "The model has low confidence in this prediction."
        }
        st.info(f"💡 {confidence_explanation[result['confidence']]}")
        
        # User features
        st.subheader("👤 User Feature Analysis")
        feature_analysis = {
            "Age": f"{age} years",
            "Tenure": f"{tenure_months} months",
            "Monthly Income": f"${income_monthly:,.0f}",
            "Savings Rate": f"{savings_rate:.1%}",
            "Risk Score": f"{risk_score}/100",
            "App Activity": f"{app_opens_7d} opens/week",
            "Session Duration": f"{avg_session_min:.1f} minutes",
            "Country": country,
            "Portfolio": f"Equity {equity_pct:.0f}% + Bond {bond_pct:.0f}% + Cash {cash_pct:.0f}% + Crypto {crypto_pct:.0f}%"
        }
        for k, v in feature_analysis.items():
            st.write(f"**{k}**: {v}")
        
        st.caption(f"Prediction Time: {result['timestamp']}")
        
        # Suggestions
        st.subheader("💡 Suggestions")
        if result["prediction"] == 1:
            st.success("🎉 This user has a high chance of converting. Consider offering personalized services.")
        else:
            st.info("📈 This user has a lower chance of converting. Consider marketing campaigns to increase engagement.")
        
        # Portfolio suggestions
        if equity_pct > 60:
            st.warning("⚠️ High equity allocation — higher risk")
        elif equity_pct < 20:
            st.info("💡 Low equity allocation — consider increasing equities")
        
        if crypto_pct > 20:
            st.warning("⚠️ High crypto allocation — very risky")
        elif crypto_pct > 0:
            st.info("💡 Crypto investment detected — inherently high risk")

# Run main program
if __name__ == "__main__":
    main()
