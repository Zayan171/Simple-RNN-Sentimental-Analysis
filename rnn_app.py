import streamlit as st
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


# Load trained model
model = load_model("simple_rnn_imdb.h5")


# Load word index
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)


# Preprocess text
def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review


# Predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(
        preprocessed_input,
        verbose=0
    )

    score = prediction[0][0]

    if score > 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, score


# Streamlit App
st.title("IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review below:")

review = st.text_area("Movie Review")


if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        sentiment, score = predict_sentiment(review)

        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success("Positive Review")
        else:
            st.error("Negative Review")

        st.write(f"Prediction Score: {score:.4f}")