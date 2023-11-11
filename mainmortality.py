# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import shap

st.header("An Artificial Intelligence Mobile Application for Predicting 30-Day Mortality in Critically Ill Patients with orthopaedic trauma")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Age = st.sidebar.slider("Age", 10, 80)
Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
Osteoporosis = st.sidebar.selectbox("Osteoporosis", ("No", "Yes"))
Hypertension = st.sidebar.selectbox("Hypertension", ("No", "Yes"))
Diabetes = st.sidebar.selectbox("Diabetes", ("No", "Yes"))
Oldmyocardialinfarction= st.sidebar.selectbox("Old myocardial infarction", ("No", "Yes"))
Firstcreatinine= st.sidebar.slider("Creatinine (mg/dL)", 0.50, 1.50)
Firstureanitrogen = st.sidebar.slider("Blood urea nitrogen (mg/dL)", 10, 35)
Redbloodcell = st.sidebar.slider("Red blood cell (m/uL)", 3.00, 5.00)
Plateletcount = st.sidebar.slider("Platelet count (K/uL)", 150, 350)
Heartrate = st.sidebar.slider("Heart rate (BPM)", 50, 125)
Resprate = st.sidebar.slider("Respiratory rate (BPM)", 8, 30)
Temperaturecelsius = st.sidebar.slider("Temperature (celsius)", 34.0, 39.0)
SAPSII = st.sidebar.slider("SAPSII", 10, 50)
SOFA = st.sidebar.slider("SOFA", 0, 10)

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round-web.pkl")
    x = pd.DataFrame([[Gender,Osteoporosis,Hypertension,Diabetes,Oldmyocardialinfarction,Age,Firstcreatinine,Firstureanitrogen,Redbloodcell,Plateletcount,Heartrate,Resprate,Temperaturecelsius,SAPSII,SOFA]],
                     columns=["Gender","Osteoporosis","Hypertension","Diabetes","Oldmyocardialinfarction","Age","Firstcreatinine","Firstureanitrogen","Redbloodcell","Plateletcount","Heartrate","Resprate","Temperaturecelsius","SAPSII","SOFA"])
    x = x.replace(["Male", "Female"], [1, 2])
    x = x.replace(["No", "Yes"], [0, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Risk of early death: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.627:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")
    if prediction < 0.627:
        st.success(f"For patients with a low early mortality risk in the ICU who have suffered orthopaedic trauma, a conservative treatment plan focused on pain management, physical therapy, and early mobilization is recommended. The primary goal is to promote patient comfort, prevent complications, and facilitate a smooth recovery process. Pain management plays a crucial role in the management of these patients. Adequate pain control is essential to ensure patient comfort and facilitate early mobilization. This may involve the use of analgesic medications, regional anesthesia techniques, or non-pharmacological interventions. Close monitoring of vital signs, pain levels, wound healing, and functional status is essential in detecting any potential complications or changes in the patient's condition. This allows for timely intervention and modification of the treatment plan if necessary. Regular follow-up appointments with orthopaedic surgeons or other healthcare providers involved in the patient's care are also scheduled to assess progress, address any concerns, and ensure continuity of care.")
    else:
        st.error(f"Recommendations: In the case of high early mortality risk, a multidisciplinary approach is essential. This includes close monitoring of vital signs, prompt identification and management of complications, and early surgical intervention when necessary. Additionally, optimizing pain control, nutritional support, and infection prevention measures are crucial in reducing mortality risk.")

    st.subheader('Model explanation: contribution of each model predictor')
    star = pd.read_csv('X_train.csv', low_memory=False)
    y_trainy = pd.read_csv('y_train.csv', low_memory=False)
    data_train_X = star.loc[:, ["Gender","Osteoporosis","Hypertension","Diabetes","Oldmyocardialinfarction","Age","Firstcreatinine","Firstureanitrogen","Redbloodcell","Plateletcount","Heartrate","Resprate","Temperaturecelsius","SAPSII","SOFA"]]
    y_train = y_trainy.Death30daysdischarge
    model = rf_clf.fit(data_train_X, y_train)
    explainer = shap.Explainer(model)
    shap_value = explainer(x)
    #st.text(shap_value)

    shap.initjs()
    #image = shap.plots.force(shap_value)
    #image = shap.plots.bar(shap_value)

    shap.plots.waterfall(shap_value[0])
    st.pyplot(bbox_inches='tight')
    st.set_option('deprecation.showPyplotGlobalUse', False)


st.subheader('About the model')
st.markdown('This online calculator is freely accessible and utilizes the advanced extreme gradient boosting machine algorithm. Validation of the model has demonstrated exceptional performance, achieving an impressive AUC of 0.974 (95%CI: 0.959-0.983). However, it is crucial to emphasize that this model was developed solely for research purposes. Therefore, clinical treatment decisions for bone metastases should not be solely reliant on the AI platform. Instead, it is recommended to consider the model’s predictions as an additional tool to aid in decision-making, complementing the expertise and judgment of healthcare professionals.')
