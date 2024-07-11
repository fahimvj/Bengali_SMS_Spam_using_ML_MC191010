from altair import Header
import streamlit as st
import pickle
import re
import string 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer


#main task started   
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.markdown('

<p>&nbsp;</p>
<h2 style="text-align: center;"><strong>A Project Proposal on</strong></h2>
<h3 style="text-align: center;"><strong>BANGLA SMS SPAM DATASET AND DETECTION USING MACHINE LEARNING</strong></h3>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p><strong>Supervised by</strong></p>
<p><strong>Mohammed Mahmudur Rahman</strong></p>
<p>Associate Professor</p>
<p>Dept. Of CSE, IIUC</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p><strong>Submitted by</strong></p>
<p><strong>MD Fahim Shahriar Chowdhury</strong></p>
<p>MC191010</p>
, unsafe_allow_html=True)

st.markdown('<span style="font-size: 12px;">by MD. Fahim Shahriar Chowdhury (MC191010)</span>', unsafe_allow_html=True)

input_sms = st.text_input("Enter the message")

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


