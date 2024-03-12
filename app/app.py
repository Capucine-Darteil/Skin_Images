import streamlit as st
import requests
import os
from PIL import Image
from Skin_Project.params import *
import numpy as np
from io import StringIO

st.title(':violet[Skin cancer] detector')

st.markdown("### Please upload your mole photo")
img_file_buffer = st.file_uploader('Upload an image')

if img_file_buffer is not None :
    st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")

if img_file_buffer is not None:
    col1, col2= st.columns(2)
    with col1:
        if st.button('Predict binary classification'):
            with st.spinner("Predict binary classification"):
                    img_bytes = img_file_buffer.getvalue()
                    res = requests.post(API_URL + "/binary_classification", files={'img': img_bytes})

                    if res.status_code == 200:
                ### Display the image returned by the API
                        final = res.content
                        st.markdown(final.decode('utf-8'))
                    else:
                        st.markdown("**Oops**, something went wrong 😓 Please try again.")
                        print(res.status_code, res.content)

    with col2:
        if st.button('Predict multiclass'):
            with st.spinner("Wait for it..."):
                        img_bytes = img_file_buffer.getvalue()
                        res = requests.post(API_URL + "/multiclass_classification", files={'img': img_bytes})

                        if res.status_code == 200:
                    ### Display the image returned by the API
                            final = res.content
                            st.markdown(final.decode('utf-8'))
                        else:
                            st.markdown("**Oops**, something went wrong 😓 Please try again.")
                            print(res.status_code, res.content)




sex = st.selectbox('Your sex:',('Male', 'Female'))
age = st.number_input('Your age:', key=int,min_value=1, max_value=99)
localization = st.selectbox('The localization of the mole:', ('Lower extremity', 'Trunk', 'Upper extremity', 'Scalp', 'Abdomen', 'Ear', 'Back', 'Face','Chest','Foot','Neck','Scalp','Hand','Genital','Acral'))



data = {
    "sex":sex,
    "age":age,
    "localization":localization
}

if st.button('Predict with metadata'):
    with st.spinner("Wait for it..."):
        img_bytes = img_file_buffer.getvalue()
        res = requests.post(API_URL + "/predict_metadata", files={'img': img_bytes},data=data)
        st.write(res)
        if res.status_code == 200:
            ### Display the image returned by the API
            final = res.content
            st.markdown(final)
                    # st.markdown(final.decode('utf-8'))
        else:
            st.markdown("**Oops**, something went wrong 😓 Please try again.")
            st.write(res.status_code, res.content)
