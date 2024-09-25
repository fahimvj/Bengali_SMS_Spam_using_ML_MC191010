from altair import Header
import streamlit as st
import pickle
import re
import string 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer


#main task started   
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.markdown('<h4 style="text-align: center;"><strong>BANGLA SMS SPAM DETECTION USING MACHINE LEARNING</strong></h4>', unsafe_allow_html=True)


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
          st.write(f'** Given Input Text: ** {input_sms}')
          st.header("\n Spam (এই মেসেজ টি সম্ভবত স্প্যাম)")
         
        else:
          st.write(f'**Given Input Text: ** {input_sms}')
          st.header("\n Not Spam (এই মেসেজ টি সম্ভবত স্প্যাম নই। )")

  
st.write('\n')         
st.write('\n ------------------------Some Sample Bengali SMS (কিছু স্যাম্পল বাংলা এস এম এস) ------------------------')
st.markdown('<p style="font-size:80%;">পূর্বাচল৩০নং সেক্টরের সাথেই ইন্ডাস্ট্রিয়াল প্লট কাঠা ১.৬লক্ষ 01894841756</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:80%;">জরুরি মুহূর্তে ১০০ টাকা পর্যন্ত ঝটপট ব্যালেন্স লোন পেতে ডায়াল *123*007# </p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:80%;">NOVOAIR-এ ব্র্যাক ব্যাংক কার্ডে ১০% ছাড়!  tinyurl.com/bblnvai</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:80%;">স্ট্যানচার্ট সিগনেচার ও প্ল্যাটিনাম ক্রেডিট এবং প্রায়োরিটি ডেবিট কার্ড ব্যবহারে Levis ফ্ল্যাগশিপ স্টোরে ১০% ছাড়। বিশদ: 01324244997</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:80%;">PayGo ইন্টারনেট সীমা অতিক্রম করেছে। নতুন ইন্টারনেট প্যাক কিনতে *121*3# ডায়াল করুন অথবা ক্লিক bit.ly/pay-g0</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:80%;">(Chaldal) ফজলি🥭৳১০৯  হাড়িভাঙ্গা আম৳১৩০ 🎁কোডঃU4 T&C chdl.co/jdd4xjzHx</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:80%;">বর্তমানে ফায়ার সার্ভিস ও সিভিল ডিফেন্স এর নতুন হটলাইন নম্বর ১০২। ১৬১৬৩ হটলাইন নম্বরটি ০১লা জানুয়ারি ২০২৫ খ্রিঃ হতে অকার্যকর থাকবে।</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:80%;">ক্লিয়ারেন্স সেল!  DEEN জিন্সে ফ্ল্যাট 30% ছাড়! ভিজিট deencommerce.com</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:80%;">প্রিয় গ্রাহক, আপনার প্যাকের মেয়াদ শীঘ্রই শেষ হয়ে যাবে, আরও আকর্ষণীয়  অফার পেতে ডায়াল *০#</p>', unsafe_allow_html=True)

