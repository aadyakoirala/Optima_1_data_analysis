import pickle
import pandas as pd
import streamlit as st

st.title("Customer Churn Prediction App")

model = pickle.load(open("xgb_model.pkl", "rb"))
encoder = pickle.load(open("my_encoder.pkl", "rb"))
## Input Fields
education = st.selectbox("Education", encoder.named_transformers_["cat"].categories_[0])
income_level = st.selectbox("Income Level", encoder.named_transformers_["cat"].categories_[1])
device_type = st.selectbox("Device Type", encoder.named_transformers_["cat"].categories_[2])
product = st.selectbox("Product", encoder.named_transformers_["cat"].categories_[3])

age = st.number_input("Age", min_value=0)
tech_comfort_score = st.number_input("Tech Comfort Score", min_value=0.0)
subscription_amount = st.number_input("Subscription Amount", min_value=0.0)
total_sessions = st.number_input("Total Sessions", min_value=0.0)
total_session_minutes = st.number_input("Total Session Minutes", min_value=0.0)
tenure_days = st.number_input("Tenure Days", min_value=0)
recency_days = st.number_input("Recency Days", min_value=0)
engagement_score = st.number_input("Engagement Score", min_value=0.0, max_value=1.0)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        "AGE": age,
        "EDUCATION": education,
        "INCOME_LEVEL": income_level,
        "DEVICE_TYPE": device_type,
        "TECH_COMFORT_SCORE": tech_comfort_score,
        "PRODUCT": product,
        "SUBSCRIPTION_AMOUNT": subscription_amount,
        "TOTAL_SESSIONS": total_sessions,
        "TOTAL_SESSION_MINUTES": total_session_minutes,
        "TENURE_DAYS": tenure_days,
        "RECENCY_DAYS": recency_days,
        "ENGAGEMENT_SCORE": engagement_score
    }])

    # Reorder columns to match encoder
    input_df = input_df[encoder.feature_names_in_]

    # Encode
    encoded = encoder.transform(input_df)

    # Predict
    prob = model.predict_proba(encoded)[0][1]
    pred = model.predict(encoded)[0]

    st.success(f"Churn Probability: {prob:.2f}")
    st.info("Prediction: Churn" if pred == 1 else "Prediction: Not Churn")
