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
        st.write(f'**Input Message:** {input_sms}')
        if result == 1:
          st.header("\n Spam (এই মেসেজ টি সম্ভবত স্প্যাম)")
         
        else:
          st.header("\n Not Spam (এই মেসেজ টি সম্ভবত স্প্যাম নই। )")

  
         

st.write('\n নিজেই প্রতিশোধ নিও না, আল্লাহর জন্য অপেক্ষা কর। তাহলে তিনি তোমাকে রক্ষা করবেন।')
st.write('\n আপনার যাচাইকরন কোডটি হল ৪৫৬৮৭৭৮')
st.write('\n কল ড্রপের জন্য আপনি রবি র পক্ষ থেকে ৩০ সেকেন্ড বোনাস পেয়েছেন।')
st.write('\n আজকে চাল-৬৫ টাকা, ডিম - ৯৯ টাকা মাত্র, এখনি কিনে ফেলুন, কুপন কোড- 3U chaldal.com')
