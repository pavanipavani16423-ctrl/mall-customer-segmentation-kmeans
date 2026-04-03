# ============================================
# Customer Sentiment Intelligence System
# Professional 3-Class Sentiment Model
# ============================================

from transformers import pipeline

# Load a better 3-class sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Label mapping
label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def predict_sentiment(text):
    """
    Predict sentiment of a given review text
    Returns:
        sentiment_label (Negative / Neutral / Positive)
        confidence_score
    """

    result = sentiment_model(text)[0]

    raw_label = result["label"]
    confidence = result["score"]

    sentiment = label_mapping.get(raw_label, raw_label)

    return sentiment, confidence