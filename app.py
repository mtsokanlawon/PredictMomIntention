import streamlit as st
import pandas as pd
import numpy as np
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import pickle 
import matplotlib.pyplot as plt
import matplotlib
import seaborn

# Import other relevant libraries for your model
from datetime import datetime

# Test data
test_data_path = "app_test_data.csv"
test_data = pd.read_csv(test_data_path)
test_data = test_data.drop(test_data.columns[0], axis=1)


# Specify the file path from which you want to load the pickled model
# file_path = 'classifier.pkl'
# file_path_encoder = 'leEncoder.pkl'
file_path = 'xgbclf.pkl'
file_path2 = 'votingclf0.pkl'
file_path_encoder = 'votinglencoder.pkl'
file_path_scaler = 'scaler.pkl'

# Load the pickled model from the file
with open(file_path, 'rb') as f:
    model = pickle.load(f)

with open(file_path2, 'rb') as f:
    model2 = pickle.load(f)

# Load the picked encoder from file
with open(file_path_encoder, 'rb') as f:
    le = pickle.load(f)

# Load pickled scaler from file
with open(file_path_scaler, 'rb') as f:
    scaler = pickle.load(f)


# Title and description
st.title("Mother's Intention to Feed Baby Prediction")
st.write("""
This app predicts how a mother intends to feed her baby using various factors. 
Please provide the information from the sidebar on left to get a prediction. Or use test data from table below!
""")

# Sidebar for inputs
st.sidebar.header('Input Parameters')

# assigning scores from 1 to 5 for prediction decoding.
ordinal_dict = {
    2:'Breast milk only',
    1:'Breast and bottle', 
    0:'Bottle milk only',  
    4:'Other',
    3:"Don't know yet"
    }

# Collecting user input through the sidebar
def user_input_features():
    WeightPrePregkg =  st.sidebar.slider("Mother's Weight before Pregnancy", 0, 200, 10)
    Heightcm = st.sidebar.slider('Mother\'s Height in Centimetre', 50, 250, 5)
    FeelStfdSprtCare = st.sidebar.slider('Support Care Satisfaction', 1, 5, 1)
    RangeCoLivers = st.sidebar.slider('Number of Co-Livers', 0, 50, 1 )
    NHseHldTalkTouch = st.sidebar.slider('Number of People Talked with and Touched in your Household (yesterday)', 0, 50, 1 )
    StartTime = st.sidebar.slider("What date is it?", 
                                   datetime(2024, 1, 1, 0, 0), 
                                   datetime(2024, 12, 30, 0, 0),  
                                   format="MM/DD/YY - hh:mm", )#
    ExpDelvryDate = st.sidebar.slider('Expected Delivery Date', 
                                      datetime(2024, 1, 1, 0, 0), 
                                      datetime(2024, 12,30, 0, 0))#
    DrinkAlcohol = st.sidebar.selectbox('Do You Drink Alcohol?',
                                       [ 'Yes, but I stopped before I was pregnant', 
                                        'Yes, I stopped as soon as I knew I was pregnant',
                                        'No, I have never drunk alcohol', 'Other',
                                        'Yes, about once per week', 'Yes, very occasionally now' 
                                            ])
    FeelNAE = st.sidebar.selectbox('Do you feel nervous or anxious?', 
                                   ['Several days', 
                                    'Not at all', 
                                    'More than half the days', 
                                    'Nearly everyday'
                                    ])
    WalkPace = st.sidebar.selectbox('Walk Pace', 
                                    ['Steady average pace', 
                                     'Brisk pace', 
                                     'Fast pace',
                                     'Slow pace']                                    )
    COVIDChangeBirthType = st.sidebar.selectbox('Did COVID change your birth type?', 
                                                ['No, it has not changed the type of birth I intend to have (home birth, midwife unit, water birth, c-section, hospital consultant led)',
                                                'Yes, my options have been removed and I do not have the choice to have the birth I would choose.',
                                                'Yes, I have changed the type of birth I intend to have (e.g. home birth, midwife unit, water birth, C-section, hospital consultant lead)',
                                                'Other', 
                                                "Don't know yet",]
                                                )
    TypeMaternCare = st.sidebar.selectbox('Type of Maternity Care', 
                                          ['Midwife led care', 'Consultant led care', 'Other'])
    HighestEduLvl = st.sidebar.selectbox('Highest Level of Education', 
                                         ['University higher degree', 'University degree',
                                        'Higher national diploma',
                                        'Exams at age 16 (GCSE or equivalent)',
                                        'Exams at age 18 (A level or equivalent)',
                                        'Vocational qualifications', 'PhD', 
                                        'Diploma of Higher Education',
                                        'NVQ', 'Postgraduate', 'Masters', 
                                        'CACHE L5', 'Undergraduate'])
    TotHseHldIncome = st.sidebar.selectbox('Total House Hold Income', 
                                           ['Prefer not to say', 
                                            'Less than 10,000',
                                            'Between 10,000 to 19,999',
                                            'Between 20,000 to 29,999', 
                                            'Between 30,000 to 39,999',
                                            'Between 40,000 to 49,999',
                                            'Above 50,000', 
                                            ])
    DeliveryDate = st.sidebar.slider('Date of Delivery (expected_delivery_date)', 
                                     datetime(2024, 1, 1, 0, 0), 
                                     datetime(2024, 12, 30, 0, 0)) #
    
    # Create a dictionary of features to pass into the model
    data = {'WeightPrePregkg':WeightPrePregkg, 
            'Heightcm':Heightcm, 
            'FeelStfdSprtCare':FeelStfdSprtCare, 
            'RangeCoLivers':RangeCoLivers, 
            'NHseHldTalkTouch':NHseHldTalkTouch, 
            "StartTime":StartTime, 
            'ExpDelvryDate':ExpDelvryDate, 
            'DrinkAlcohol':DrinkAlcohol, 
            'FeelNAE':FeelNAE, 
            'WalkPace':WalkPace, 
            # 'How do Plan to Feed your Baby?':PlanFeedBaby,      
            'COVIDChangeBirthType':COVIDChangeBirthType, 
            'TypeMaternCare':TypeMaternCare, 
            'HighestEduLvl':HighestEduLvl, 
            'TotHseHldIncome':TotHseHldIncome, 
            'DeliveryDate':DeliveryDate
       }
    
    # Return the input as a pandas dataframe
    features = pd.DataFrame(data, index=[0])
    return features

# Store user input
input_df = user_input_features()

# Display the input parameters for verification
st.subheader('**User Input Parameters**')
st.write('**<<<** Select Input From Side Bar')
st.write(input_df)

# Encode categorical inputs
cat_inputs = input_df.select_dtypes(["object", "datetime"]).columns

for cat_var in input_df[cat_inputs]:
    input_df[cat_var] = le.fit_transform(input_df[cat_var],)

input_scaled = scaler.transform(input_df)


# model prediction for input
prediction1 = model.predict(input_scaled)
prediction2 = model2.predict(input_scaled)

decoded_prediction1 = ordinal_dict[prediction1[0]]
decoded_prediction2 = ordinal_dict[prediction2[0]]
# use the model to predict

prediction_proba = model.predict_proba(input_df)
prediction_proba_df = pd.DataFrame(prediction_proba)
prediction_proba_df.rename(columns = ordinal_dict, inplace=True)

decoded_prediction_proba = prediction_proba_df

if st.button('Predict!', type='primary'):
    # model prediction output
    st.subheader('Prediction')
    st.write('Using Model XGB...')
    st.write(f"Based on the input data, the mother is predicted to feed by: **{decoded_prediction1}**")
    st.write('Using Model Voting...')
    st.write(f"Based on the input data, the mother is predicted to feed by: **{decoded_prediction2}**")


if st.button('Show XGB Prediction Probability!'): 
    # Display prediction probabilities
    st.subheader('Prediction Probability')
    st.write(decoded_prediction_proba)

    labels = decoded_prediction_proba.columns
    sizes = decoded_prediction_proba.values[0]
    explode = [1,1,1,1,0.0]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, explode=explode,
        autopct='%1.1f%%', startangle=180
        )
    ax.axis('equal')
    st.pyplot(fig)


st.subheader('Perform Prediction on Test Data')
st.write('**!Select a row of choice and click Predict Selection Below.**')
def data_selection(df):
    event = st.dataframe(test_data, on_select="rerun", 
                         key="data", hide_index=True,
                         selection_mode="single-row")
    
    selection_df = event.selection
    return selection_df

# Or select a test data row from below

selection_dict = data_selection(test_data)
selection_df = test_data.iloc[selection_dict["rows"]]
st.write('**Selected Test Data**')
st.write(selection_df)

# Hide error details from the user
st.set_option('client.showErrorDetails', False)

# Clean the features for prediction
selection_df = selection_df.drop(["PlanFeedBaby", "PlanFeedNum"], axis=1)

# Encoding selection
cat_selection = selection_df.select_dtypes(["object", "datetime"]).columns

for cat_var in selection_df[cat_selection]:
    selection_df[cat_var] = le.fit_transform(selection_df[cat_var],)

selection_scaled = scaler.transform(selection_df)

prediction = model.predict(selection_scaled)
prediction0 = model2.predict(selection_scaled)
 
decoded_prediction = ordinal_dict[prediction[0]]
decoded_prediction0 = ordinal_dict[prediction0[0]]

prediction_prob = model.predict_proba(selection_df)
prediction_prob_df = pd.DataFrame(prediction_prob)
prediction_prob_df.rename(columns = ordinal_dict, inplace=True)

decoded_prediction_prob = prediction_prob_df

if st.button("Predict Selection", type='primary'):
    # model prediction output
    st.subheader('Prediction')
    st.write('Using Model XGB...')
    st.write(f"Based on the input data, the mother is predicted to feed by: **{decoded_prediction}**")
    st.write('Using Model Voting...')
    st.write(f"Based on the input data, the mother is predicted to feed by: **{decoded_prediction0}**")
    
if st.button('Show Prediction Probability'):
    # Display prediction probabilities
    st.subheader('Prediction Probability')
    st.write(decoded_prediction_prob)

    labels = decoded_prediction_prob.columns
    sizes = decoded_prediction_prob.values[0]
    explode = [1,1,1,1,0.0]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, explode=explode,
        autopct='%1.1f%%', startangle=180
        )
    ax.axis('equal')
    st.pyplot(fig)


