
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("simple_rnn_imdb.keras")

with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [1]

    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2)

    return pad_sequences([encoded_review], maxlen=500)


def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict(processed_review, verbose=0)
    score = float(prediction[0][0])

    if score > 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, score


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