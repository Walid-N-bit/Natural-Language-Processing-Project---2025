import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from task_2 import penn_to_wn_tag
from task_8 import raw_entities, damage_sentences
import csv
import os

wnl = nltk.WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()


def sentiment_polarity(raw_text: str):
    """
    compute polarity as VADER compound score.
    the score ranges from -1 (most negative) to +1 (most positive)
    """
    sents_tokens = sent_tokenize(raw_text)
    compound = []
    for sent in sents_tokens:
        s_score = sid.polarity_scores(sent)
        compound.append(s_score["compound"])
    return sum(compound) / len(compound)


# emotion intensity
def emotion_intensity(raw_text: str):
    sents_tokens = sent_tokenize(raw_text)
    negative = []
    neutral = []
    positive = []
    for sent in sents_tokens:
        s_score = sid.polarity_scores(sent)
        negative.append(s_score["neg"])
        neutral.append(s_score["neu"])
        positive.append(s_score["pos"])
    return {
        "neg": sum(negative) / len(negative),
        "neu": sum(neutral) / len(neutral),
        "pos": sum(positive) / len(positive),
    }


# damage-related words frequency
def damage_frequency(text: str):
    summary_size = len(damage_sentences(text))
    return summary_size


def impact_score(emo_intensity: dict, senti_polarity: float, dam_freq: float):
    """ """
    norm_polarity = (senti_polarity + 1) / 2
    neg_intensity = emo_intensity["neg"]
    damage = min(dam_freq, 20)
    norm_damage = damage / 20
    score = (norm_polarity + norm_damage + neg_intensity) / 3
    return score


# ranking articles
def impact2csv(in_data: list[list[str]], out_path: str = "data/scored_articles.csv"):
    """
    take input data from csv file, compute impact score for each article
    save to a new file with impact scores
    """
    file_exists = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    out_fields = [
        "article_title",
        "source",
        "neg_emo_intensity",
        "neu_emo_intensity",
        "pos_emo_intensity",
        "sentiment_polarity",
        "damage_frequency",
        "impact_score",
    ]
    out_data = []

    for row in in_data[1:]:
        title, _, source, article_text, clean_text = row
        polarity = sentiment_polarity(clean_text)
        intensity = emotion_intensity(article_text)
        neg = emotion_intensity(article_text)["neg"]
        neu = emotion_intensity(article_text)["neu"]
        pos = emotion_intensity(article_text)["pos"]
        damage = damage_frequency(article_text)
        imp_score = impact_score(intensity, polarity, damage)
        out_data.append([title, source, neg, neu, pos, polarity, damage, imp_score])

    sorted_data = sorted(out_data, key=lambda x: x[5])
    file_exists = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    with open(out_path, mode="a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(out_fields)
        csvwriter.writerows(sorted_data)
    return sorted_data


# # testing

from task_2 import load_file

data = load_file("data/cleaned_data.csv")

scored_articles = impact2csv(in_data=data)

# print(scored_articles)

# # polarity = sentiment_polarity(text[0][4])
# # intensity = emotion_intensity(text[0][3])
# # damage = damage_frequency(text[0][3])
# # impact = impcat_score(polarity, intensity, damage)
# # print("sentiment polarity = ", polarity)
# # print("emotion intensity = ", intensity)
# # print("damage freq = ", damage)
# # print("impact score = ", impact)
