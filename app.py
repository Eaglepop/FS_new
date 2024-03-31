# first we imported libraries we need
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
import xgboost 
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib

# import streamlit.web.cli 
# from streamlit.web.cli import main
import sys

# sys.path.append('/opt/homebrew/anaconda3/envs/venv38/lib/python3.8/site-packages')

# Create a app title with title method
st.title('Post-DCSD')

# st.markdown('<p style= "font-size": 100px;>Select Model                                 Infection Caculator</p>', unsafe_allow_html=True)

# We called back our models created before
# model1 =pickle.load(open("ada_xgb.pkl","rb"))
# model1 = joblib.load("ada_xgb.pkl")
# model2= pickle.load(open("ada_xgb.pkl","rb"))

# with open('I_Models/ros_xgb_i.pkl', 'rb') as f:
#     model1 = pickle.load(f)

# with open('/Users/isadmin/Desktop/研究所課程/fs_streamlit/I_Models/ros_rf_i.pkl', 'rb') as f:
#     model2 = pickle.load(f)

# with open('I_Models/ros_lgbm_i.pkl', 'rb') as f:
#     model3 = pickle.load(f)

# with open('/I_Models/smt_xgb_i.pkl', 'rb') as f:
#     model4 = pickle.load(f)

# with open('/I_Models/smt_rf_i.pkl', 'rb') as f:
#     model5 = pickle.load(f)

# with open('/I_Models/smt_lgbm_i.pkl', 'rb') as f:
#     model6 = pickle.load(f)
    
# with open('/I_Models/smtnc_xgb_i.pkl', 'rb') as f:
#     model7 = pickle.load(f)

# with open('/I_Models/smtnc_rf_i', 'rb') as f:
#     model8 = pickle.load(f)

# with open('/I_Models/smtnc_lgbm_i.pkl', 'rb') as f:
#     model9 = pickle.load(f)

# with open('/I_Models/smttom_xgb_i.pkl', 'rb') as f:
#     model10 = pickle.load(f)

# with open('/I_Models/smttom_rf_i', 'rb') as f:
#     model11 = pickle.load(f)

# with open('/I_Models/smttom_lgbm_i.pkl', 'rb') as f:
#     model12 = pickle.load(f)

# with open('/I_Models/ada_xgb_i.pkl', 'rb') as f:
#     model13 = pickle.load(f)

# with open('/I_Models/ada_rf_i', 'rb') as f:
#     model14 = pickle.load(f)

# with open('/I_Models/ada_lgbm_i.pkl', 'rb') as f:
#     model15 = pickle.load(f)
    





with open('ros_xgb_i.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('ros_rf_i.pkl', 'rb') as f:
    model2 = pickle.load(f)

with open('ros_lgbm_i.pkl', 'rb') as f:
    model3 = pickle.load(f)

with open('smt_xgb_i.pkl', 'rb') as f:
    model4 = pickle.load(f)

with open('smt_rf_i.pkl', 'rb') as f:
    model5 = pickle.load(f)

with open('smt_lgbm_i.pkl', 'rb') as f:
    model6 = pickle.load(f)
    
with open('smtnc_xgb_i.pkl', 'rb') as f:
    model7 = pickle.load(f)

with open('smtnc_rf_i.pkl', 'rb') as f:
    model8 = pickle.load(f)

with open('smtnc_lgbm_i.pkl', 'rb') as f:
    model9 = pickle.load(f)

with open('smttom_xgb_i.pkl', 'rb') as f:
    model10 = pickle.load(f)

with open('smttom_rf_i.pkl', 'rb') as f:
    model11 = pickle.load(f)

with open('smttom_lgbm_i.pkl', 'rb') as f:
    model12 = pickle.load(f)

with open('ada_xgb_i.pkl', 'rb') as f:
    model13 = pickle.load(f)

with open('ada_rf_i.pkl', 'rb') as f:
    model14 = pickle.load(f)

with open('ada_lgbm_i.pkl', 'rb') as f:
    model15 = pickle.load(f)
    
    
    
    
    
    
    
    
    
# We use selectbox method and append our models to give a choice clients
# models = st.selectbox("Select Model",("ros_xgb","ros_rf","ros_lgbm",
#                                       "smt_xgb","smt_rf","smt_lgbm",
#                                       "smtnc_xgb","smtnc_rf","smtnc_lgbm",
#                                       "smttom_xgb","smttom_rf","smttom_lgbm",
#                                       "ada_xgb","ada_rf","ada_lgbm") )

st.markdown('##')

c1, c2 = st.columns([3,1])
c1.subheader('Select model')
c2.subheader('Infection caculator')

models = st.selectbox('',("xgb","rf") )

# And specified a condition if users select Random forest use random forest model else use Xgboost model.
if models == "xgb":
    model = model13
elif models == "rf":
    model = model14
# if models == "ros_xgb":
#     model = model1
# elif models == "ros_rf":
#     model = model2
# elif models == "ros_lgbm":
#     model = model3
# elif models == "smt_xgb":
#     model = model4
# elif models == "smt_rf":
#     model = model5
# elif models == "smt_lgbm":
#     model = model6
# elif models == "smtnc_xgb":
#     model = model7
# elif models == "smtnc_rf":
#     model = model8
# elif models == "smtnc_lgbm":
#     model = model9
# elif models == "smttom_xgb":
#     model = model10
# elif models == "smttom_rf":
#     model = model11
# elif models == "smttom_lgbm":
#     model = model12
# elif models == "ada_xgb":
#     model = model13
# elif models == "ada_rf":
#     model = model14
# elif models == "ada_lgbm":
#     model = model15


# We created selectbox for categorical columns and used slider numerical values ,specified range and step
# age = st.selectbox("What is the age of your car?",(1,2,3))
# hp = st.slider("What is the horsepower of your car",60,200,step=5)
# km=st.slider("What is the km of your car?",0,100000,step=500)
# car_model=st.selectbox("Select model of your car", ('A1', 'A2', 'A3','Astra','Clio','Corsa','Espace','Insignia'))
st.markdown('<p> </p>', unsafe_allow_html=True)
st.markdown('<p> </p>', unsafe_allow_html=True)
st.markdown('<p style= "font-size": 100px;>Infection Caculator</p>', unsafe_allow_html=True)


Age = st.slider("Age",1,100,step=1)
Height = st.slider("Height",50,250,step=1)
Weight = st.slider("Weight",0,200,step=1)
Penis_length = st.slider("Penis's length",1,30,step=1)
Phimosis_Grading = st.slider("Phimosis grading",0,5,step=1)
Pain_level_D0 = st.slider("Pain level day 0",0,10,step=1)
Pain_level_D1 = st.slider("Pain level day 1",0,10,step=1)
Re_inflammation = st.selectbox("Re inflammation",('Yes','No'))
Diabetes = st.selectbox("Diabetes",('Yes','No'))
Tilting = st.selectbox("Tilting",('Yes','No'))
Foreskin_edema = st.selectbox("Foreskin edema",('Yes','No'))
Surg_time = st.slider("Surgery time",0,100,step=1)
Tool_type = st.selectbox("Tool type",('22','24','26','30','34'))
# in order to recieved client inputs appended these inputs (created above) into dictionary as we mentioned before. And We returned into dataframe.
my_dict = {
    'Age': Age, 'Height': Height, 'Weight': Weight,
    'Penis_length': Penis_length, 'Phimosis_Grading': Phimosis_Grading,
    'Pain_level_D0': Pain_level_D0, 'Pain_level_D1': Pain_level_D1,
    'Re_inflammation': Re_inflammation, 'Diabetes': Diabetes,
    'Tilting': Tilting, 'Foreskin_edema': Foreskin_edema,
    'Surg_time': Surg_time, 'Tool_type': Tool_type
          }

columns = ['Age', 'Height', 'Weight', 'Penis_length', 'Phimosis_Grading',
       'Pain_level_D0', 'Pain_level_D1', 'Re_inflammation',
       'Diabetes', 'Tilting', 'Foreskin_edema','Surg_time','Tool_type']


def predict():
    # row = np.array([Age, Height, Weight, Penis_length, Phimosis_Grading,
    #    Pain_level_D0, Pain_level_D1, Re_inflammation,
    #    Diabetes, Tilting, Foreskin_edema,Surg_time,Tool_type])
    df = pd.DataFrame.from_dict([my_dict])
    # df = pd.DataFrame([my_dict],columns=columns)
    df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    probability=np.array(probability)
    print(probability[1])
    print(model.predict(df)[0])
     
    if prediction == 1:
        # success='<p color:Green; style= "font-size": 100px;>Infected</p>'
        # st.markdown(success, unsafe_allow_html=True)
        # st.markdown('<p color:Green; style= "font-size": 100px;>Infected probability: probability[1]</p>', unsafe_allow_html=True)


        # st.success('<p style= "font-size": 50px;>Infected</p>')
        st.success('Infected')
        
        st.write('Infected probability:', probability[1])
    else:
        # error='<p color:Red; style= "font-size": 100px;>NotInfected</p>'
        # st.markdown(error, unsafe_allow_html=True)

        
        # st.error('<p style= "font-size": 50px;>Not Infected</p>')
        st.error('Not Infected')
        
        
        st.write('Infected probability:', probability[1])
        
   
    
    

st.button('Predict', on_click=predict)

# df = pd.DataFrame.from_dict([my_dict])
# # And appended column names into column list. We need columns to use with reindex method as we mentioned before.
# df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
# # We append all columns in the user input dataframe and reindex method just received relevant user inputs , and return other columns from nan to zero with fill_value=0 parameter.
# # And now we can predict
# print(df.index)
# prediction = model.predict(df)




# # first we imported libraries we need
# import streamlit as st
# import pickle
# import pandas as pd
# # Create a app title with title method
# st.title('Foreskin Prediction')
# # We called back our models created before
# model1 =pickle.load(open("xgb_model","rb"))
# model2= pickle.load(open("rf_model","rb"))
# # We use selectbox method and append our models to give a choice clients
# models = st.selectbox("Select Model",("Random Forest","XGBoost","LightGBM") )
# # And specified a condition if users select Random forest use random forest model else use Xgboost model.
# if models == "Random Forest":
#     model = model2
# else :
#     model = model1
# # We created selectbox for categorical columns and used slider numerical values ,specified range and step 
# # age = st.selectbox("What is the age of your car?",(1,2,3))
# # hp = st.slider("What is the horsepower of your car",60,200,step=5)
# # km=st.slider("What is the km of your car?",0,100000,step=500)
# # car_model=st.selectbox("Select model of your car", ('A1', 'A2', 'A3','Astra','Clio','Corsa','Espace','Insignia'))

# Age = st.slider("Age",1,100,step=1)
# Height = st.slider("Height",50,250,step=1)
# Weight = st.slider("Weight",0,200,step=1)
# Penis_length = st.slider("Penis's length",1,30,step=1)
# Phimosis_Grading = st.slider("Phimosis grading",0,5,step=1)
# Pain_level_D0 = st.slider("Pain level day 0",0,10,step=1)
# Pain_level_D1 = st.slider("Pain level day 1",0,10,step=1)
# Re_inflammation = st.selectbox("Re inflammation",('Yes','No'))
# Diabetes = st.selectbox("Diabetes",('Yes','No'))
# Tilting = st.selectbox("Tilting",('Yes','No'))
# Foreskin_edema = st.selectbox("Foreskin edema",('Yes','No'))
# Surg_time = st.slider("Surgery time",0,100,step=1)
# Tool_type = st.selectbox("Tool type",('22','24','26','30','34'))
# # in order to recieved client inputs appended these inputs (created above) into dictionary as we mentioned before. And We returned into dataframe.
# my_dict = {
#     'Age': Age, 'Height': Height, 'Weight': Weight, 
#     'Penis_length': Penis_length, 'Phimosis_Grading': Phimosis_Grading,
#     'Pain_level_D0': Pain_level_D0, 'Pain_level_D1': Pain_level_D1,
#     'Re_inflammation': Re_inflammation, 'Diabetes': Diabetes,
#     'Tilting': Tilting, 'Foreskin_edema': Foreskin_edema,
#     'Surg_time': Surg_time, 'Tool_type': Tool_type
#           }
# df = pd.DataFrame.from_dict([my_dict])
# # And appended column names into column list. We need columns to use with reindex method as we mentioned before.
# df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
# # We append all columns in the user input dataframe and reindex method just received relevant user inputs , and return other columns from nan to zero with fill_value=0 parameter.
# # And now we can predict
# prediction = model.predict(df)
# # Success method demonstrate our prediction in a green square
# st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))