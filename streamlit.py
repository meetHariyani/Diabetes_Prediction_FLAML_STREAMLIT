import joblib  # For loading pickle file.
import pandas as pd # For dataframe usage
import streamlit as st #to create Streamlit Webapp.


# Adding elements to webapp.

st.header("Diabetes priction using ML model")


age = st.number_input('Insert age of the patient.',min_value=1, max_value=100,value=int(1),step=1)

gender = st.radio(
     'Gender ',
     ['MALE','FEMALE'])

if gender == 'MALE':
     gender = 1
else:
     gender = 0

polyuria = st.radio(
     "patient experiences excessive urination ?",
     ('Yes', 'No'))

if polyuria == 'Yes':
     polyuria = 1
else:
     polyuria = 0

polydipsia = st.radio(
     "Patient experiences excessive thirst/excess drinking ?",
     ('Yes', 'No'))

if polydipsia == 'Yes':
     polydipsia = 1
else:
     polydipsia = 0

sudden_weight_loss = st.radio(
     "Patient had an episode of sudden weight loss ?",
     ('Yes', 'No'))

if sudden_weight_loss == 'Yes':
     sudden_weight_loss = 1
else:
     sudden_weight_loss = 0

weakness = st.radio(
     "Patient had an episode of feeling weaks ?",
     ('Yes', 'No'))

if weakness == 'Yes':
     weakness = 1
else:
     weakness = 0

polyphagia= st.radio(
     "Patient had an episode of excessive/extreme hunger ?",
     ('Yes', 'No'))

if polyphagia == 'Yes':
     polyphagia = 1
else:
     polyphagia = 0

genital_thrush = st.radio(
     "Patient had a yeast infection on genitals ?",
     ('Yes', 'No'))

if genital_thrush == 'Yes':
     genital_thrush = 1
else:
     genital_thrush = 0

visual_blurring = st.radio(
     "Patient had an episode of blurred vision ?",
     ('Yes', 'No'))

if visual_blurring == 'Yes':
     visual_blurring = 1
else:
     visual_blurring = 0

itching = st.radio(
     "Patient had an episode of itch ?",
     ('Yes', 'No'))

if itching == 'Yes':
     itching = 1
else:
     itching = 0

irritability = st.radio(
     "Patient had an episode of irritating feeling ?",
     ('Yes', 'No'))

if irritability == 'Yes':
     irritability = 1
else:
     irritability = 0

delayed_healing = st.radio(
     "Patient had an episode of delayed healing ?",
     ('Yes', 'No'))

if delayed_healing == 'Yes':
     delayed_healing = 1
else:
     delayed_healing = 0

partial_paresis = st.radio(
     "Patient is having weakness of voluntary movement ?",
     ('Yes', 'No'))

if partial_paresis == 'Yes':
     partial_paresis = 1
else:
     partial_paresis = 0

muscle_stiffness = st.radio(
     "Patient is feeling mucle stiffness ?",
     ('Yes', 'No'))

if muscle_stiffness == 'Yes':
     muscle_stiffness = 1
else:
     muscle_stiffness = 0

alopecia = st.radio(
     "Patient is having sudden hair loss problem ?",
     ('Yes', 'No'))

if alopecia == 'Yes':
     alopecia = 1
else:
     alopecia = 0

obesity = st.radio(
     "Patient is having excessive body fat ?",
     ('Yes', 'No'))

if obesity == 'Yes':
     obesity = 1
else:
     obesity = 0



# storing input data in dictionary.
predictInput = {
    'age' : int(age),
    'gender': gender,
    'polyuria' : polyuria,
    'polydipsia' : polydipsia,
    'sudden_weight_loss' : sudden_weight_loss,
    'weakness' : weakness,
    'polyphagia' : polyphagia,
    'genital_thrush':genital_thrush,
    'visual_blurring':visual_blurring,
    'itching':itching,
    'irritability':irritability,
    'delayed_healing':delayed_healing,
    'partial_paresis':partial_paresis,
    'muscle_stiffness':muscle_stiffness,
    'alopecia':alopecia,
    'obesity':obesity
}

# Converting Input data into pandas dataframe 
input = pd.DataFrame(predictInput,index=[0])

# Loading pickle file.
MLmodel = joblib.load('https://github.com/meetHariyani/Diabetes_Prediction_FLAML_STREAMLIT/blob/b13272fd1566f65acacaad874b4fd2a5f8262fd9/model_local.pkl
')

# Predicting result using pre trained model.
prediction = MLmodel.predict(input).astype(str)

# Showing prediction.
if st.button('Predict'):
     if prediction == ['1']:
          st.write('Diabetic')
     else:
          st.write('non-diabetic')
