import streamlit as st
import requests
import os
from PIL import Image
from Skin_Project.params import *
import numpy as np
from io import StringIO

st.title(':violet[Skin cancer] detector')
st.title('_Streamlit_ is :red[cool]')

backgroundColor="#FFFFFF"

with st.form(key='params_for_api'):
    gender = st.selectbox('Your gender:',('Male', 'Female'))
    age = st.number_input('Your age:', key=int,min_value=1, max_value=99)
    location = st.selectbox('The localization of the mole:', ('Lower extremity', 'Trunk', 'Upper extremity', 'Scalp', 'Abdomen', 'Ear', 'Back', 'Face','Chest','Foot','Neck','Scalp','Hand','Genital','Acral'))
    st.form_submit_button("Submit")

st.markdown("### Please upload your mole photo")
img_file_buffer = st.file_uploader('Upload an image')

if img_file_buffer is not None:
    col1, col2 = st.columns(2)
    with col1:
    ### Display the image user uploaded
        st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")
    with col2:
        with st.spinner("Wait for it..."):
            img_bytes = img_file_buffer.getvalue()
            res = requests.post(API_URL + "/upload_image", files={'img': img_bytes})

            if res.status_code == 200:
        ### Display the image returned by the API
                final = res.content
                st.markdown(final.decode('utf-8'))
            else:
                st.markdown("**Oops**, something went wrong 😓 Please try again.")
                print(res.status_code, res.content)
