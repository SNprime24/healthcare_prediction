import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import os


st.title("HEALTH DETERMINATION MODEL 🧑‍⚕️")

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Home","Health Status","About"],
        icons  = ["house","file-earmark-medical","bookmark-star-fill"],
        default_index=0,
        # orientation="horizontal"
    )

script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current script's directory
csv_path = os.path.join(script_dir, 'health.csv') # Construct the relative path to the CSV file
df = pd.read_csv(csv_path) # Read the CSV file into a DataFrame


def bmi_calculator():
    st.subheader("BMI CALCULATOR")
    with st.form('BMI Calculator'):
        col1,col2,col3 = st.columns([2,2,1])
    with col1:
        weight = st.number_input ('Enter your weight (in **kgs**)')
    with col2:
        height = st.number_input ('Enter your height (in **metres**)')
    with col3:
        submit_BMI = st.form_submit_button('Calculate')

    if submit_BMI:
        BMI = round((weight/height**2),2)
        st.markdown(f"#### Your BMI is {BMI}")
        if(BMI <= 18.5):            
            st.error('Underweight')
        elif(BMI <=24.9):
            st.success('Healthy/Normal')
        elif(BMI <= 29.9):
            st.warning('Overweight')
        else:
            st.error('Obese')
        return BMI

if(selected=="Home"):
    st.header("HOME 🏠")
    st.markdown("###### Welcome to the Health Predictions Web Application, your personalized health assistant! Our advanced neural network is here to help you assess potential health risks based on your input. Whether you're concerned about heart disease, skin cancer, or other conditions, our tool provides accurate predictions to guide your health decisions.")
    st.subheader("**Key Features**")
    st.markdown("- ##### *Personalized Predictions :*")
    st.markdown("Input your health report, and our neural network will generate personalized predictions tailored to your profile.")
    st.markdown("- ##### *Trained Neural Network :*")
    st.markdown("Our system has been trained on a diverse dataset from Kaggle, ensuring accurate and reliable predictions.")
    st.markdown("- ##### *User-Friendly Interface :*")
    st.markdown("Easily navigate the application with our intuitive and user-friendly web interface.")
    st.subheader("**GET STARTED**")
    st.markdown("Ready to explore your health predictions? Click on the 'Health Status' button on the sidebar to input your health information and receive personalized predictions instantly. Your well-being is our priority!")
    
if(selected=="Health Status"):
    st.header("HEALTH STATUS 🗃️")
    BMI=bmi_calculator()
    st.subheader("HEALTH REPORT")
    with st.form('Health Form'):
        your_bmi = st.number_input('Enter your BMI (Calculate from the Top):', value = 0,)
        your_gender = st.radio('Select your Gender :',options ={'Male','Female','Other'}, help = 'Choose One',horizontal = True )
        your_agecategory = st.selectbox('Select your Age Category :',options =df['AgeCategory'].sort_values().unique(), help = 'Choose One' )
        your_race = st.radio('Select your Race :',options =df['Race'].unique(), help = 'Choose One',horizontal = True )
        col1,col2,col3 = st.columns([1,1,1])
        with col1:
            your_smoke = st.radio('Do you Smoke :',options =df['Smoking'].unique(), help = 'Choose One',horizontal = True,index=1 )
        with col2:
            your_drink = st.radio('Do you Drink :',options =df['Smoking'].unique(), help = 'Choose One',horizontal = True, index=1 )
        with col3:
            your_stroke = st.radio('Did you get any stroke :',options =df['Smoking'].unique(), help = 'Choose One',horizontal = True, index=1 )
        col4,col5 =st.columns([1,1])
        with col4:
            your_walk = st.radio('Do you feel difficulty in walking :',options =df['Smoking'].unique(), help = 'Choose One',horizontal = True, index=1 )
        with col5:
            your_phyactivity = st.radio('Do you do any physical activity :',options =df['Smoking'].unique(), help = 'Choose One',horizontal = True, index=1 )
        your_diabetes = st.radio('Are you suffering from Diabetes :',options =df['Diabetic'].unique(), help = 'Choose One',horizontal = True, index=1 )
        your_sleep_time = st.slider('What is your average sleep time ?', min_value=0, max_value=24, value=8)
        your_phealth = st.slider('Rate your Physical Health : ', min_value=0, max_value=100, value=0)
        your_mhealth = st.slider('Rate your Mental Health : ', min_value=0, max_value=100, value=0)
        your_ghealth = st.slider('what is your expection with reference to General Health : ', min_value=0, max_value=4, value=2)
        submit_BMI = st.form_submit_button('Submit the Report')
    
    if(submit_BMI):
        
        your_smoke = 1 if your_smoke == "Yes" else 0
        your_drink = 1 if your_drink == "Yes" else 0
        your_stroke = 1 if your_stroke == "Yes" else 0
        
        q1 = df['PhysicalHealth'].quantile(0.25)
        q3 = df['PhysicalHealth'].quantile(0.75)
        iqr = q3-q1
        your_phealth = q1+(your_phealth/100)*iqr
        
        q1 = df['MentalHealth'].quantile(0.25)
        q3 = df['MentalHealth'].quantile(0.75)
        iqr = q3-q1
        your_mhealth = q1+(your_mhealth/100)*iqr

        your_walk = 1 if your_walk == "Yes" else 0
        your_gender = 1 if your_gender == "Male" else 0

        agecategory_column = df['AgeCategory'].sort_values().unique()
        for i in range(len(agecategory_column)):
            if(agecategory_column[i]==your_agecategory):
                your_agecategory=i

        Race_column = ['Black','White','Asian','American Indian/Alaskan Native','Hispanic','Other']
        for i in range(len(Race_column)):
            if(Race_column[i]==your_race):
                your_race=i

        diabetes_column = df['Diabetic'].unique()
        for i in range(len(diabetes_column)):
            if(diabetes_column[i]==your_diabetes):
                your_diabetes=i

        your_phyactivity = 1 if your_phyactivity == "Yes" else 0

        lst=[your_bmi,your_smoke,your_drink,your_stroke,your_phealth,your_mhealth,your_walk,your_gender,your_agecategory,your_race,your_diabetes,your_phyactivity,your_ghealth,your_sleep_time]
        case=np.array(lst);
        input_data_reshaped = np.expand_dims(case, axis=0)

        model_asthma = tf.keras.models.load_model('model_asthama.h5')
        asthama_y_pred = model_asthma.predict(input_data_reshaped)
        asthama_chances=round((asthama_y_pred[0,0]*100),2)        

        model_cancer = tf.keras.models.load_model('model_cancer.h5')
        cancer_y_pred = model_cancer.predict(input_data_reshaped)
        cancer_chances=round((cancer_y_pred[0,0]*100),2)
        
        model_heart = tf.keras.models.load_model('model_heart.h5')
        heart_y_pred = model_heart.predict(input_data_reshaped)
        heart_chances=round((heart_y_pred[0,0]*100),2)

        model_kidney = tf.keras.models.load_model('model_kidney.h5')
        kidney_y_pred = model_kidney.predict(input_data_reshaped)
        kidney_chances=round((kidney_y_pred[0,0]*100),2)

        with st.expander("See Report"):
            st.markdown(f"### Chances of getting Asthama in Future :  {asthama_chances} %")
            st.markdown(f"### Chances of getting Heart Disease in Future : {heart_chances} %")
            st.markdown(f"### Chances of getting Skin Cancer in Future : {cancer_chances} %")
            st.markdown(f"### Chances of getting kidney Disease in Future : {kidney_chances} %")
            


if(selected=="About"):
    st.header("ABOUT 🙏")
    st.subheader('Our Mission :')
    st.markdown("At Health Predictions, our mission is to empower individuals with valuable health insights. We believe that informed decisions lead to healthier lives. By combining user aided technology with data-driven predictions, we aim to make health assessments accessible to everyone.")
    st.subheader('Neural Network Training :')
    st.markdown("Our neural network has undergone rigorous training using a curated dataset sourced from Kaggle. This diverse dataset enables our system to recognize patterns and correlations, providing you with accurate and reliable predictions.")
    st.subheader('Data Source :')
    st.markdown("Data taken from Kaggle-2022 annual CDC survey data of 400k+ adults related to their health status")
    st.dataframe(df)
    st.subheader("Team:")
    st.markdown("- 👨‍💻 Makarandh - 22CS01002")
    st.markdown("- 👨‍💻 Suprit Naik - 22CS01018")
    st.markdown("- 👨‍💻 Harsh Maurya - 22CS01046")
