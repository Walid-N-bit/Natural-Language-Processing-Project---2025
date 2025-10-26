import nltk
from nltk.corpus import stopwords
import csv

STOPWORDS = stopwords.words("english")
DEFAULT_DATA_PATH = "data/articles.csv"


def load_file(path: str = DEFAULT_DATA_PATH):
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)

        return [row for row in reader]


def is_alpha_not_sw(word: str):
    return word.isalpha() and word not in STOPWORDS


def clean_text(data: list[list[str]]):
    data = data[1:]
    new_data = []
    for row in data:
        text = row[3].split()
        clean_text_words = [w.lower() for w in text if is_alpha_not_sw(w.lower())]
        clean_text = " ".join(clean_text_words)
        new_data.append([*row, clean_text])

    return new_data



# # testing
# data = load_file()
# clean_text(data)
