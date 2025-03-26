import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
import keras
import streamlit as st

### opening tarined model 

model = tf.keras.models.load_model('model.h5')

####opening encoders and scalers 

with open('lable_encoder_gender.pkl', 'rb') as f:
    lable_encoder_gender = pickle.load(f)

with open ('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler  = pickle.load(f)


#### streamlit app configuration 

st.title('customer churn prediction ')

####user input

CreditScore = st.number_input('credit_score')        
Geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
Gender = st.selectbox('gender',lable_encoder_gender.classes_ )            
Age = st.slider('age', 17, 93)
Tenure = st.slider('tenure', 0, 10)
Balance = st.number_input('balance')
NumOfProducts = st.slider('nnum_products', 0,10)
HasCrCard = st.selectbox('credit_card', [0,1])
IsActiveMember = st.selectbox('is active??', [0,1])
EstimatedSalary = st.number_input('est salary')




####prepare input data

input_data = pd.DataFrame({

    'CreditScore': [CreditScore],
    'Gender': [lable_encoder_gender.transform([Gender])[0]], 
    'Age' : [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})


geo_encoder = one_hot_encoder_geo.transform([[Geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoder, columns=one_hot_encoder_geo.get_feature_names_out())

input_data = pd.concat([input_data.reset_index(drop = True), geo_encoder_df], axis = 1)



input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
predict_proba = prediction[0][0]

st.write(f'churn probability:{predict_proba:.2f}')


if predict_proba > 0.5:
    st.write('the customer is likely a churn')
else:
    st.write('the customer is not likely a churn')