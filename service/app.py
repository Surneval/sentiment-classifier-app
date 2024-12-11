# service/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_inference import SentimentModel

app = FastAPI(title="IMDB Sentiment Classifier API")

# initialize the sentiment model
model = SentimentModel(model_path="./model")

class Review(BaseModel):
    review: str

@app.post("/predict", summary="Predict the sentiment of a movie review")
def predict_sentiment(review: Review):
    if not review.review.strip():
        raise HTTPException(status_code=400, detail="Review text is empty.")
    sentiment = model.predict(review.review)
    return {"sentiment": sentiment}

