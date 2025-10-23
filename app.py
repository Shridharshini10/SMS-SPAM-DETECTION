import streamlit as st
import pickle
from preprocessing import clean_text

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("📱 SMS Spam Detector")
msg = st.text_area("Enter your message")
if st.button("Predict"):
    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    st.write("🚫 Spam" if pred[0] else "✅ Not Spam")
