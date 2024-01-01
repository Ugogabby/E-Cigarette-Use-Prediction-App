import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle

############################################################################################### Page Setup #################################################################################################### 

st.set_page_config(
    page_title="E-Cigarrete Use Prediction App",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "This is an E-Cigarrete Use Prediction App!"}
)

st.write(
  """## :blue[Welcome to the E-Cigarrete Use Prediction App]

  **This server implements the :rainbow[Random Forest Model] for predicting the potential of E-cigarette use among veterans and Non-Veterans in the US.  
    The data for the model training was obtained from the publicly Available _US NHANES_.**
  """
)

st.divider()

####################################################################################### Load Model and Coded-data ############################################################################################## 

try:
  with open("rfc.pkl", mode="rb") as infile1, open("codified_imput.json", mode="rt") as infile2:
    rfc = pickle.load(infile1)
    coded_data = json.load(infile2)
except:
  st.error("### An error occured while trying to load data, Refresh to reload data")

################################################################################################# SideBar ###################################################################################################### 
with st.sidebar:
  st.write("""<p style="text-align:justify">Paste your data and click the <b>Predict</b> button to predict your E-Cigarrete Use Potential</p>""", unsafe_allow_html=True)
  name = st.text_input("Name (:red[Optional])", placeholder="Name")
  gender = st.radio("Gender of the participant (:red[Required])", ["Male", "Female"], index=None, disabled=False, horizontal=True, captions=None, label_visibility="visible")
  age = st.number_input("Age in years (:red[Required])", min_value=0, max_value=150, value=None, step=5, placeholder="Enter your age", disabled=False, label_visibility="visible")
  race_and_hispanic_origin = st.selectbox("Race/Hispanic origin (:red[Required])", ["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Other Race - Including Multi-Racial"], 
                                          index=None, placeholder="Choose the race that best describes you", disabled=False, label_visibility="visible")
  country_of_birth = st.radio("In what country were you/was SP born (:red[Required])", ["Born in 50 US states or Washington, DC", "Others"], 
                                          index=None, disabled=False, horizontal=True, captions=None, label_visibility="visible")
  marital_status = st.radio("Marital status (:red[Required])", ["Married/Living with Partner", "Widowed/Divorced/Separated", "Never married"], index=None, disabled=False, horizontal=True, captions=None, label_visibility="visible")
  veteran_status = st.radio("Veteran Status (:red[Required])", ["Veteran", "Non-Veteran"], index=None, disabled=False, horizontal=True, captions=None, label_visibility="visible")
  highest_education_grade_received = st.selectbox("Highest grade or level of school you have/SP has completed? (:red[Required])", ["Less than 9th grade", "9-11th grade (Includes 12th grade with no diploma)", "High school graduate/GED or equivalent", "Some college or AA degree", "College graduate or above"], 
                                                   index=None, placeholder="Choose the Educational grade that best suites you", disabled=False, label_visibility="visible")
  button = st.button("**Predict**")


####################################################################################### Class Prediction ############################################################################################## 

# Define a style function to left-align text
def left_align_text(val):
    return "text-align: left"

# Convert dataframe to csv file for download
def convert_df(df):
  return df.to_csv().encode('utf-8')

if button:
    try:
      name_ = name if name is not None else "Classified"
      df = pd.DataFrame({'gender': [coded_data["gender"][gender]], 'age(year)': [age], 'race_and_hispanic_origin': [coded_data["race_and_hispanic_origin"][race_and_hispanic_origin]], 
                        'country_of_birth': [coded_data["country_of_birth"][country_of_birth]], 'marital_status': [coded_data["marital_status"][marital_status]], 'veteran_status': [coded_data["veteran_status"][veteran_status]], 
                        'highest_education_grade_received': [coded_data["highest_education_grade_received"][highest_education_grade_received]]})
      
      probabilities = rfc.predict_proba(df.values.reshape(1, -1))
      predicted_class = np.argmax(probabilities, axis=1)
      probability = probabilities[0, predicted_class]
      class_ = "Did Not Smoke" if predicted_class[0] == 0 else "Smoke"
      
      result = pd.DataFrame({"Name": name_, "Gender": gender, "Age (Year)": age, "Race/Hispanic origin": race_and_hispanic_origin,
                            "Country of birth": country_of_birth, "Marital status": marital_status, "Veteran status": veteran_status,
                            "Highest education level": highest_education_grade_received, "Predicted class": class_, "Probability": probability[0]},
                            index=["Data"])
      
      result = result.T
      result.index.name = "Personal Information"

      # styled_result = result.style.applymap(left_align_text)
      # pd.set_option('display.max_colwidth', None)
      # pd.set_option('display.precision', 2)

      st.write(result)

      csv = convert_df(result)

      st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='participants_data.csv',
        mime='text/csv',)
    except:
      st.error("### :red[Something went wrong. All required details must be filled]")

else:
  st.write("### :gray[Your predictions will appear here]")

st.divider()

####################################################################################### Model Training Information ############################################################################################## 
  
st.write("### :grey[Given below are the model trainining data]")
col1, col2, col3 = st.columns(3)
with col1:
  st.metric("#### :rainbow[Accuracy]", "97.91 %")
with col2:
  st.metric("#### :rainbow[AUC]", "99.00 %")
with col3:
  st.metric("#### :rainbow[F1-Score]", "98.07 %")

st.divider()

st.image("model_AUC.png")

st.divider()
