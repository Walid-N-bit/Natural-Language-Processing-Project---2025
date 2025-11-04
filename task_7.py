import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline


# Initializing models
print("\nInitializing models...")

# VADER para sentiment
vader_analyzer = SentimentIntensityAnalyzer()
print("VADER sentiment analyzer is loaded")

# Hugging Face para emotions
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=-1,  # CPU
)
print("Emotion classifier is loaded")

# Functions


def analyze_sentiment(text):

    scores = vader_analyzer.polarity_scores(text)
    return {
        "vader_compound": scores["compound"],
        "vader_positive": scores["pos"],
        "vader_negative": scores["neg"],
        "vader_neutral": scores["neu"],
    }


def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"


def analyze_emotions(text, max_length=512):
    words = text.split()
    if len(words) > max_length:
        text = " ".join(words[:max_length])

    try:
        emotions = emotion_classifier(text)[0]
        return {e["label"]: e["score"] for e in emotions}
    except Exception as e:
        print(f"   Error: {e}")
        return {
            "anger": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "joy": 0.0,
            "neutral": 0.0,
            "sadness": 0.0,
            "surprise": 0.0,
        }


def get_dominant_emotion(emotion_dict):
    """Obtener emoción dominante"""
    return max(emotion_dict, key=emotion_dict.get)


# Complete Analysis


def comprehensive_analysis(df):
    print("\n🔍 Analyzing articles...")
    print("   Using VADER for sentiment + Transformers for emotions")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = row["clean_text"]

        try:
            # VADER Sentiment
            sentiment_scores = analyze_sentiment(text)
            sentiment_label = classify_sentiment(sentiment_scores["vader_compound"])

            # Hugging Face Emotions
            emotions = analyze_emotions(text)
            dominant_emotion = get_dominant_emotion(emotions)

            # Derived metrics
            emotion_intensity = max(emotions.values())
            fear_sadness_ratio = (
                emotions["fear"] / emotions["sadness"]
                if emotions["sadness"] > 0.01
                else 0
            )

            # Combine results
            article_data = {
                "article_id": idx,
                "title": row["title"],
                "date": row.get("date", "N/A"),
                "source": row.get("source", "N/A"),
                # Sentiment (VADER)
                **sentiment_scores,
                "sentiment_label": sentiment_label,
                # Emotions (Hugging Face)
                **emotions,
                "dominant_emotion": dominant_emotion,
                # Derived metrics
                "emotion_intensity": emotion_intensity,
                "fear_sadness_ratio": fear_sadness_ratio,
                # Negative emotion sum
                "negative_emotion_sum": emotions["fear"]
                + emotions["sadness"]
                + emotions["anger"],
            }

            results.append(article_data)

        except Exception as e:
            print(f"\nError in article {idx}: {e}")
            continue

    return pd.DataFrame(results)


# Testing

# # Load DataFrame
# df = pd.read_csv(r".\data\cleaned_data.csv")
# print(f"\nLoaded {len(df)} articles")


# # Execute the analysis

# sentiment_df = comprehensive_analysis(df)

# # Saved
# sentiment_df.to_csv(r".\data\sentiment_emotion_analysis.csv", index=False)
# print(f"Results saved to 'sentiment_emotion_analysis.csv'")

# # Statistics

# print("\n" + "=" * 70)
# print("DESCRIPTIVE STATISTICS")
# print("=" * 70)

# # Sentiment
# print("\nSENTIMENT ANALYSIS (VADER):")
# sentiment_cols = ["vader_compound", "vader_positive", "vader_negative", "vader_neutral"]
# print(sentiment_df[sentiment_cols].describe())

# # Emotions
# print("\nEMOTION DETECTION (Hugging Face Transformer):")
# emotion_cols = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
# print(sentiment_df[emotion_cols].describe())

# # Important things

# print("\n" + "=" * 70)
# print("KEY FINDINGS")
# print("=" * 70)

# # Average sentiment
# avg_sentiment = sentiment_df["vader_compound"].mean()
# print(f"\n Average Sentiment (VADER): {avg_sentiment:.3f}")
# if avg_sentiment < -0.3:
#     print("   - Highly negative ")
# elif avg_sentiment < -0.1:
#     print("   - Moderately negative")
# else:
#     print("   - Neutral/Mixed coverage")

# # Top 3 emotions
# print("\nTop 3 Emotions (by average intensity):")
# avg_emotions = sentiment_df[emotion_cols].mean().sort_values(ascending=False)
# for i, (emotion, score) in enumerate(avg_emotions.head(3).items(), 1):
#     print(f"   {i}. {emotion.capitalize():10s}: {score:.3f}")

# # Fear/Sadness analysis
# fear_avg = sentiment_df["fear"].mean()
# sadness_avg = sentiment_df["sadness"].mean()
# fs_ratio = sentiment_df["fear_sadness_ratio"].mean()

# print(f"\nFear vs Sadness Analysis:")
# print(f"   Fear average:    {fear_avg:.3f}")
# print(f"   Sadness average: {sadness_avg:.3f}")
