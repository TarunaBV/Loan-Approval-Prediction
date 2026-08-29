import streamlit as st
import numpy as np
import pickle

with open('lin_reg_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Loan Approval Prediction System")
st.write("Enter Applicant's details below:")

edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
if edu == "Graduate":
    edu = 1
else:
    edu = 0
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
if self_employed == "Yes":
    self_employed = 1
else:
    self_employed = 0
dep = st.selectbox("Number of Dependents", ["0", "1", "2", "3", "4", "5", "5+"])
if dep == "5+":
    dep = 5
else:
    dep = int(dep)
ann_inc = st.number_input("Applicant's Annual Income", min_value=0)
loan_amt = st.number_input("Loan Amount", min_value=0)
term = st.number_input("Loan Term (in months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900)
residential_asset_val = st.number_input("Residential Asset Value", min_value=0)
comm_asset_val = st.number_input("Commercial Asset Value", min_value=0)
lux_asset_val = st.number_input("Luxury Asset Value", min_value=0)
bank_asset_val = st.number_input("Bank Asset Value", min_value=0)

if st.button("Predict"):
    input_data = np.array([[edu, self_employed, dep, ann_inc, loan_amt, term, cibil_score,
                            residential_asset_val, comm_asset_val, lux_asset_val, bank_asset_val]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.success("The loan is likely to be approved.")
    else:
        st.error("The loan is likely to be rejected.")
