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


        #1 Process
         #cleaning text

        #2 Vectorize
        sms_vector = tfidf.transform([input_sms])

        #converting to dense array
        sms_vector_dense =  sms_vector.toarray()

        #3 Predict
        result = model.predict(sms_vector_dense)[0]

        #4 Display
        if result == 1:
          st.header("\n Spam")
        else:
          st.header("\n Not Spam")


