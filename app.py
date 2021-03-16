import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings

st.set_page_config(page_title="conorary Heart disease Prediction", page_icon="💉", layout='centered', initial_sidebar_state='auto')

# Data columns
feature_names_best = ['age', 'gender', 'hypertension','hpgrade', 'diabete', 'smokecurrent', 'chestpain','cho', 'rccach', 'max_ccach', 'max_ccacl', 'ps']


gender_dict = {"Male":1,"Female":0}
feature_dict = {"Yes":1,"No":0}
feature2_dict ={"normal":0,"prehypertension":0.5,"grade 1":1,"grade2":2,"grade 3":3}
feature3_dict ={"Asymptomatic":0,"Non-anginal pain":1,"Atypical angina":2,"Typical angina":3}

def load_image(img):
	im =Image.open(os.path.join(img))
	return im

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_fvalue(val):
	feature_dict = {"Yes":1,"No":0}
	for key,value in feature_dict.items():
		if val == key:
			return value 

def get_hvalue(val):
	feature2_dict ={"normal":0,"prehypertension":0.5,"grade 1":1,"grade2":2,"grade 3":3}
	for key,value in feature2_dict.items():
		if val == key:
			return value 

def get_cvalue(val):
	feature3_dict ={"Asymptomatic":0,"Non-anginal pain":1,"Atypical angina":2,"Typical angina":3}
	for key,value in feature3_dict.items():
		if val == key:
			return value 

# title
html_temp = """
<div>
<h1 style="color:crimson;text-align:left;">Prediction of coronary heart disease in Patients </h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

if st.checkbox("Information"):
	'''
	Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
	Heart failure is a common event caused by CVDs. Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.
	People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
	'''
'''
## How does it work ❓ 
Complete all the questions and the machine learning model will predict the probility of patients with coronary heart disease
'''

# Logo
# st.sidebar.image('real-heart-icon-png-.png', width=120)
st.sidebar.title("Prediction Form📋")

# Age of the patient
age = st.sidebar.number_input("age", 1,100)
# Male or Female 
gender = st.sidebar.radio("gender",tuple(gender_dict.keys()))
# Increase of hypertension
hypertension = st.sidebar.radio("hypertension ?",tuple(feature_dict.keys()))
# hypertension grade
hpgrade = st.sidebar.radio("hypertension grade?",tuple(feature2_dict.keys()))
# If the patient has diabete
diabete = st.sidebar.selectbox("Diabete",tuple(feature_dict.keys()))
# If the patient smokes
smokecurrent = st.sidebar.radio("Smoking current?",tuple(feature_dict.keys()))
# If the patient chest pain
chestpain = st.sidebar.radio("chest pain",tuple(feature3_dict.keys()))
# Level of the cho in the blood
cho = st.sidebar.number_input("cholesterol (mmol/L)",0,100)
# Level of the rccach in the blood
rccach = st.sidebar.number_input("right carotid plaque height (cm)",0,10)
# Level of the max_ccach in the blood
max_ccach = st.sidebar.number_input("maximum double carotid plaque height (cm)",0,10)
# Level of the max_ccacl in the blood
max_ccacl = st.sidebar.number_input("maximum double carotid plaque length (cm)",0,10)
# Level of the ps in the blood
ps = st.sidebar.number_input("add double carotid plaque height (cm)",0,10)

feature_list = [age,get_value(gender,gender_dict),get_fvalue(hypertension),get_hvalue(hpgrade),get_fvalue(diabete),get_fvalue(smokecurrent,),get_cvalue(chestpain),cho,rccach,max_ccach,max_ccacl,ps]
pretty_result = {"age":age,"gender":gender,"hypertension ?":hypertension,"hypertension grade?":hpgrade, "Diabete":diabete,"Smoking current?":smokecurrent,"chest pain":chestpain,"cholesterol (mmol/L)" :cho, "right carotid plaque height (cm)":rccach, "maximum double carotid plaque height (cm)":max_ccach,"maximum double carotid plaque length (cm)":max_ccacl,"add double carotid plaque height (cm)":ps}
'''
## These are the values you entered 🧑‍⚕
'''
st.json(pretty_result)
single_sample = np.array(feature_list).reshape(1,-1)

if st.button("Predict"):
		'''
		## Results 👁‍🗨

		'''
		loaded_model = load_model('model.pkl')
		prediction = loaded_model.predict(single_sample)
		pred_prob = loaded_model.predict_proba(single_sample)
		
		if prediction == 1:
			st.error("The patient has a coronary heart disease")
		else:
			st.success("The patient hasn't a coronary heart disease")
			
		for cad, no_cad in loaded_model.predict_proba(single_sample):
				cad = f"{cad*100:.2f}%"
				no_cad = f"{no_cad*100:.2f} %"
				st.table(pd.DataFrame({'cad ':cad,
								'no_cad': no_cad}, index=['probability']))
				st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon")

st.sidebar.subheader("Source code")
st.sidebar.info('''

[![Github](https://i.ibb.co/vDLv9z9/iconfinder-mark-github-298822-3.png)](https://github.com/baox13/cadp)
**Github**
''')    

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
