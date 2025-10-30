import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from task_2 import penn_to_wn_tag
from task_8 import raw_entities, damage_summary
import csv
import os

wnl = nltk.WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()


# sentiment polarity
def sentiment_polarity(cleaned_text: str):
    pos_sentiment = 0
    neg_sentiment = 0
    sentiment_count = 0
    cleaned_tokens = word_tokenize(cleaned_text)
    tagged_tokens = pos_tag(cleaned_tokens)
    for word, tag in tagged_tokens:
        wn_tag = penn_to_wn_tag(tag)
        word_synsets = wn.synsets(word, pos=wn_tag)
        if word_synsets:
            sentiment = swn.senti_synset(word_synsets[0].name())
            if sentiment.pos_score() + sentiment.neg_score() > 0:
                pos_sentiment += sentiment.pos_score()
                neg_sentiment += sentiment.neg_score()
                sentiment_count += 1
    if sentiment_count == 0:
        return 0
    else:
        return (pos_sentiment - neg_sentiment) / sentiment_count


# emotion intensity
def emotion_intensity(raw_text: str):
    sents_tokens = sent_tokenize(raw_text)
    negative = 0
    neutral = 0
    positive = 0
    compound = 0
    for sent in sents_tokens:
        s_score = sid.polarity_scores(sent)
        negative += s_score["neg"]
        neutral += s_score["neu"]
        positive += s_score["pos"]
        compound += s_score["compound"]
    return {"neg": negative, "neu": neutral, "pos": positive, "compound": compound}


# damage-related words frequency
def damage_frequency(text: str):
    summary_size = damage_summary(text, print_sum=False)
    return summary_size


# impcat score
def impcat_score(senti_polarity: float, emo_intensity: dict, dam_freq: float):
    """
    calculated by taking the average of sentiment polarity and intensity over number of sentences describing damage
    """
    emo_compouned = emo_intensity.get("compound")
    if dam_freq == 0:
        return 0
    else:
        return (senti_polarity + emo_compouned) / (2 * dam_freq)


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
        "sentiment_polarity",
        "emotion_intensity",
        "damage_frequency",
        "impact_score",
    ]
    out_data = []

    for row in in_data[1:]:
        title, _, source, article_text, clean_text = row
        polarity = sentiment_polarity(clean_text)
        intensity = emotion_intensity(article_text)
        damage = damage_frequency(article_text)
        imp_score = impcat_score(polarity, intensity, damage)
        out_data.append([title, source, polarity, intensity, damage, imp_score])

    sorted_data = sorted(out_data, key=lambda x: x[5])
    file_exists = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    with open(out_path, mode="a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(out_fields)
        csvwriter.writerows(sorted_data)
    return sorted_data


# # testing

# from task_2 import load_file

# data = load_file("data/cleaned_data.csv")

# scored_articles = impact2csv(in_data=data)

# print(scored_articles)

# # polarity = sentiment_polarity(text[0][4])
# # intensity = emotion_intensity(text[0][3])
# # damage = damage_frequency(text[0][3])
# # impact = impcat_score(polarity, intensity, damage)
# # print("sentiment polarity = ", polarity)
# # print("emotion intensity = ", intensity)
# # print("damage freq = ", damage)
# # print("impact score = ", impact)
