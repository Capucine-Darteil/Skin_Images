import streamlit as st
import requests


with st.form(key='params_for_api'):
    gender = st.selectbox('Your gender:',('Male', 'Female'))
    age = st.number_input('Your age:', key=int)
    location = st.selectbox('The location of the mole:', ('scalp', 'abdomen', 'ear', 'back', 'face'))

response = requests.get('api', params='params')
