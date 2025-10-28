import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import csv

STOPWORDS = stopwords.words("english")
DEFAULT_DATA_PATH = "data/articles.csv"

wnl = nltk.WordNetLemmatizer()


def load_file(path: str = DEFAULT_DATA_PATH):
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)

        return [row for row in reader]


def is_alpha_not_sw(word: str):
    return word.isalpha() and word not in STOPWORDS


def penn_to_wn_tag(tag: str):

    if tag.startswith("J"):
        return wn.ADJ
    elif tag.startswith("N"):
        return wn.NOUN
    elif tag.startswith("R"):
        return wn.ADV
    elif tag.startswith("V"):
        return wn.VERB
    return wn.NOUN


def clean_text(data: list[list[str]]):
    data = data[1:]
    new_data = []
    for row in data:
        title, date, source, article_text = row
        clean_words = []
        text_tokens = word_tokenize(article_text.lower())
        tagged_tokens = pos_tag(text_tokens)
        for word, tag in tagged_tokens:
            if is_alpha_not_sw(word):
                wn_pos = penn_to_wn_tag(tag)
                lem_word = wnl.lemmatize(word, pos=wn_pos)
                clean_words.append(lem_word)
        clean_text = " ".join(clean_words)
        new_data.append([title, date, source, article_text, clean_text])

    return new_data


# # testing
data = load_file()
cleaned_data = clean_text(data)

from task_1 import articles2csv

articles2csv(
    articles=cleaned_data,
    path="data/cleaned_data.csv",
    fields=["title", "date", "source", "article_text", "clean_text"],
)
# print(load_file(path="data/cleaned_data.csv")[1][4][3])
