import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from task_2 import penn_to_wn_tag
from task_8 import raw_entities, damage_summary

wnl = nltk.WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()


# sentiment polarity
def sentiment_polarity(cleaned_text: str):
    pos_sentiment = 0
    neg_sentiment = 0
    obj_sentiment = 0
    cleaned_tokens = word_tokenize(cleaned_text)
    tagged_tokens = pos_tag(cleaned_tokens)
    for word, tag in tagged_tokens:
        wn_tag = penn_to_wn_tag(tag)
        word_synsets = wn.synsets(word, pos=wn_tag)
        if word_synsets:
            sentiment = swn.senti_synset(word_synsets[0].name())
            pos_sentiment += sentiment.pos_score()
            neg_sentiment += sentiment.neg_score()
            obj_sentiment += sentiment.obj_score()

    return pos_sentiment - neg_sentiment


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
    entities = raw_entities(text)
    summary_size = damage_summary(entities, print_sum=False)
    return summary_size


# impcat score
def impcat_score(data):
    pass


# # testing

# from task_2 import load_file

# text = load_file("data/cleaned_data.csv")[1:]
# for row in text:
#     article_sentiment_ploarity = sentiment_polarity(row[4])
#     print(f'Article "{row[0]}"\'s sentiment polarity score is: ')
#     print(article_sentiment_ploarity)

# print("emotion intensity = ", emotion_intensity(text[0][3]))
# print("damage freq = ", damage_frequency(text[0][3]))
