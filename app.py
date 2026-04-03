import streamlit as st
from train_model import predict_sentiment

st.set_page_config(page_title="Customer Sentiment Intelligence System")

st.title("📊 Customer Sentiment Intelligence System")
st.write("Analyze product or service feedback using Transformer Model")

review = st.text_area("Enter Customer Review")

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment, confidence = predict_sentiment(review)

        st.subheader("🔍 Prediction Result")
        st.success(f"Predicted Sentiment: {sentiment}")
        st.write(f"Confidence Score: {confidence:.4f}")