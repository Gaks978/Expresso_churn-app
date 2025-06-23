import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("churn_model.pkl")

st.title("Expresso Churn Prediction App")
st.markdown("Fill in client behavior details to predict churn.")

# Input fields for all model features
REGION = st.number_input("REGION (Encoded as integer)", min_value=0)
TENURE = st.number_input("TENURE (Months with the company)", min_value=0)
MONTANT = st.number_input("MONTANT (Average recharge amount)", min_value=0.0)
FREQUENCE_RECH = st.number_input("FREQUENCE_RECH (Recharge frequency)", min_value=0.0)
REVENUE = st.number_input("REVENUE", min_value=0.0)
ARPU_SEGMENT = st.number_input("ARPU_SEGMENT", min_value=0.0)
FREQUENCE = st.number_input("FREQUENCE", min_value=0.0)
DATA_VOLUME = st.number_input("DATA_VOLUME (MB used)", min_value=0.0)
ON_NET = st.number_input("ON_NET", min_value=0.0)
ORANGE = st.number_input("ORANGE", min_value=0.0)
TIGO = st.number_input("TIGO", min_value=0.0)
ZONE1 = st.number_input("ZONE1", min_value=0.0)
ZONE2 = st.number_input("ZONE2", min_value=0.0)
MRG = st.number_input("MRG", min_value=0.0)
REGULARITY = st.number_input("REGULARITY", min_value=0.0)
TOP_PACK = st.number_input("TOP_PACK (Encoded as integer)", min_value=0)
FREQ_TOP_PACK = st.number_input("FREQ_TOP_PACK", min_value=0.0)

# Button to make prediction
if st.button("Predict Churn"):
    input_data = pd.DataFrame([[
        REGION, TENURE, MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT,
        FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO, ZONE1, ZONE2, MRG,
        REGULARITY, TOP_PACK, FREQ_TOP_PACK
    ]], columns=[
        "REGION", "TENURE", "MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT",
        "FREQUENCE", "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2", "MRG",
        "REGULARITY", "TOP_PACK", "FREQ_TOP_PACK"
    ])

    prediction = model.predict(input_data)[0]
    result = "Client is likely to churn" if prediction == 1 else "Client is not likely to churn"

    st.subheader("Prediction Result:")
    st.success(result)
