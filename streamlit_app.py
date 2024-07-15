from altair import Header
import streamlit as st
import pickle
import re
import string 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer


#main task started   
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.markdown('<p>&nbsp;</p>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;"><strong>A Project Proposal on</strong></h3>', unsafe_allow_html=True)
st.markdown('<h4 style="text-align: center;"><strong>BANGLA SMS SPAM DETECTION USING MACHINE LEARNING</strong></h4>', unsafe_allow_html=True)
st.markdown('<p>&nbsp;</p>', unsafe_allow_html=True)

st.markdown('<h5><strong>Supervised by</strong></h5>', unsafe_allow_html=True)
st.markdown('<p><strong>Mohammed Mahmudur Rahman</strong></p>', unsafe_allow_html=True)
st.markdown('<p>Associate Professor, Dept of CSE, IIUC</p>', unsafe_allow_html=True)

st.markdown('<h5><strong>Submitted by</strong></h5>', unsafe_allow_html=True)
st.markdown('<p><strong>MD Fahim Shahriar Chowdhury, ID-MC191010</strong></p>', unsafe_allow_html=True)



input_sms = st.text_input("Enter the SMS Text ")

if st.button("Predict"):
    if input_sms.strip() != "":
        # Clean the input text (if necessary)
        cleaned_sms = clean_text(input_sms)

        # Vectorize the input text
        sms_vector = tfidf.transform([cleaned_sms])

        # Predict using the model
        result = model.predict(sms_vector.toarray())[0]

        # Display the input message and prediction result
        st.write(f'**Input Message:** {input_sms}')
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.error("Please enter a valid SMS text.")
         

 
