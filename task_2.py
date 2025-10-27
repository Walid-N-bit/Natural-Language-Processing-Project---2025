import nltk
from nltk.corpus import stopwords
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


def clean_text(data: list[list[str]]):
    data = data[1:]
    new_data = []
    for row in data:
        text = row[3].split()
        clean_text_words = [
            wnl.lemmatize(w.lower()) for w in text if is_alpha_not_sw(w.lower())
        ]
        clean_text = " ".join(clean_text_words)
        new_data.append([*row, clean_text])

    return new_data


# # testing
# data = load_file()
# cleaned_data = clean_text(data)

# from task_1 import articles2csv

# articles2csv(
#     articles=cleaned_data,
#     path="data/cleaned_data.csv",
#     fields=["title", "date", "source", "article_text", "clean_text"],
# )
# print(load_file(path="data/cleaned_data.csv")[1][4][3])
