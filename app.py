import streamlit as st
import pandas as pd
import joblib


model = joblib.load("LR_heart_disease_model.pkl")
expected_columns = joblib.load("columns.pkl")
scaler = joblib.load("scaler.pkl")


if 'HeartDisease' in expected_columns:
    expected_columns = [col for col in expected_columns if col != 'HeartDisease']


st.title('Heart Disease Detector with Linear Regression')
st.markdown('Provide the following details to predict your risk:')


age = st.slider('Age', 18, 100, 40)
resting_bp = st.slider('RestingBP', 80, 200, 120)
cholesterol = st.slider('Cholesterol', 0, 600, 200)
fasting_bs = st.selectbox('FastingBS', [0, 1])
max_hr = st.slider('MaxHR', 60, 220, 150)
oldpeak = st.slider('Oldpeak', -3.0, 7.0, 0.0, step=0.1)
sex = st.selectbox('Sex', ['M', 'F'])
chest_pain_type = st.selectbox('ChestPainType', ['ASY', 'ATA', 'NAP', 'TA'])
resting_ecg = st.selectbox('RestingECG', ['Normal', 'LVH', 'ST'])
exercise_angina = st.selectbox('ExerciseAngina', ['N', 'Y'])
st_slope = st.selectbox('ST_Slope', ['Up', 'Flat', 'Down'])


if st.button('Predict'):
    
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_F': 1 if sex == 'F' else 0,
        'Sex_M': 1 if sex == 'M' else 0,
        'ChestPainType_ASY': 1 if chest_pain_type == 'ASY' else 0,
        'ChestPainType_ATA': 1 if chest_pain_type == 'ATA' else 0,
        'ChestPainType_NAP': 1 if chest_pain_type == 'NAP' else 0,
        'ChestPainType_TA': 1 if chest_pain_type == 'TA' else 0,
        'RestingECG_LVH': 1 if resting_ecg == 'LVH' else 0,
        'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
        'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
        'ExerciseAngina_N': 1 if exercise_angina == 'N' else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == 'Y' else 0,
        'ST_Slope_Down': 1 if st_slope == 'Down' else 0,
        'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
        'ST_Slope_Up': 1 if st_slope == 'Up' else 0,
    }
    
    
    input_df = pd.DataFrame([raw_input])

    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    
    
    prediction = model.predict(scaled_input)
    st.markdown("---")
    st.subheader('Prediction Result:')
    
    if prediction[0] == 1:
        st.error('**Warning:** The model predicts a **High Risk** of Heart Disease. Please consult a healthcare professional.')
    else:
        st.success('**Good News:** The model predicts a **Low Risk** of Heart Disease. Keep up the healthy lifestyle!')