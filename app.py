import streamlit as st
from googleapiclient.discovery import build
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load Model
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    rating = torch.argmax(probs).item() + 1

    if rating <= 2:
        sentiment = "Negative"
    elif rating == 3:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    return sentiment, rating

st.title("📊 YouTube Sentiment Analysis")

api_key = st.text_input("Enter YouTube API Key", type="password")
video_id = st.text_input("Enter YouTube Video ID")

if st.button("Analyze"):
    if api_key == "" or video_id == "":
        st.warning("Please enter API key and Video ID")
    else:
        youtube = build("youtube", "v3", developerKey=api_key)

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=50
        )

        response = request.execute()

        pos = neg = neu = 0

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            sentiment, stars = get_sentiment(comment)

            if sentiment == "Positive":
                pos += 1
            elif sentiment == "Negative":
                neg += 1
            else:
                neu += 1

            st.write(f"📝 {comment}")
            st.write(f"➡️ {sentiment} ({stars}⭐)")
            st.write("---")

        st.subheader("Sentiment Summary")
        st.write("Positive:", pos)
        st.write("Negative:", neg)
        st.write("Neutral:", neu)
