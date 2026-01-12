from googleapiclient.discovery import build
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime

# ----------------------------
# Load Sentiment Model
# ----------------------------
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    rating = torch.argmax(probs).item() + 1   # 1 to 5 stars
    confidence = probs[0][rating-1].item()

    if rating <= 2:
        sentiment = "Negative"
    elif rating == 3:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    return sentiment, rating, round(confidence,2)

# ----------------------------
# YouTube API
# ----------------------------
API_KEY = "ENTER YOUR API KEY HERE"   # paste your key here (do NOT share)
youtube = build("youtube", "v3", developerKey=API_KEY)

video_id = " ENTER VIDEO ID"   # iphone 17 review (huge comments)

# ----------------------------
# Fetch Comments
# ----------------------------
request = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    maxResults=20
)

response = request.execute()

# ----------------------------
# Save to Text File
# ----------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"youtube_sentiment_{timestamp}.txt"

file = open(filename, "w", encoding="utf-8")

file.write("YOUTUBE SENTIMENT ANALYSIS REPORT\n")
file.write("="*50 + "\n\n")

print("\nSaving results to:", filename, "\n")

for item in response["items"]:
    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
    sentiment, stars, confidence = get_sentiment(comment)

    output = f"Comment: {comment}\nSentiment: {sentiment} | Rating: {stars}⭐ | Confidence: {confidence}\n\n"

    print(output)
    file.write(output)

file.close()

print("\nFile saved successfully ✔")
