import spacy
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import re

nlp = spacy.load("en_core_web_sm")


def aggregate_text(data: list[list[str]]):
    """
    merge the article_text columns in a dataset into one string
    """
    text = ""
    for row in data:
        text = f"{text} {row[3]}"
    return text


def raw_entities(text: str):
    """
    extract entities from a string of text
    """
    doc = nlp(text)
    return doc.ents


def extract_named_entities(text: str):

    doc = nlp(text)
    named_entities = {}
    for ent in doc.ents:
        label = ent.label_
        if named_entities.get(label) == None:
            named_entities.update({ent.label_: [ent.text]})
        else:
            named_entities.update({ent.label_: [*named_entities[ent.label_], ent.text]})
    return named_entities


def plot_entity_frequency(entities: dict):
    ents = list(entities.keys())
    freq = [len(entities[v]) for v in ents]

    # plt.figure(figsize=(10, 6))
    plt.bar(ents, freq)
    plt.xlabel("Entities")
    plt.ylabel("Frequencies")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        "entity_frequency.png"
    )  # for testing the script directly, comment-out when using jupyter notebook
    # plt.show()    # for use with jupyter notebook, comment out when running script directly with python


def is_distance(ent_text: str):
    """
    determine if a sentence contains a unit of speed or distance measurement
    """
    unit_pattern = re.compile(
        r"\d+\s*(km/h|mph|m/s|meters?|kilometers?|miles?|kms|feet|foot|inches?|centimeters?|cm|m)\b",
        re.IGNORECASE,
    )
    if unit_pattern.search(ent_text):
        return True
    else:
        return False


def damage_summary(entities, print_sum: bool = True):
    summary = set()
    for ent in entities:
        if ent.label_ in ["CARDINAL", "MONEY"] and not is_distance(ent.text):
            summary.add(ent.sent)
    if print_sum:
        for sent in summary:
            print(sent)
    return len(summary)


"""
PERSON:      People, including fictional.
NORP:        Nationalities or religious or political groups.
FAC:         Buildings, airports, highways, bridges, etc.
ORG:         Companies, agencies, institutions, etc.
GPE:         Countries, cities, states.
LOC:         Non-GPE locations, mountain ranges, bodies of water.
PRODUCT:     Objects, vehicles, foods, etc. (Not services.)
EVENT:       Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART: Titles of books, songs, etc.
LAW:         Named documents made into laws.
LANGUAGE:    Any named language.
DATE:        Absolute or relative dates or periods.
TIME:        Times smaller than a day.
PERCENT:     Percentage, including ”%“.
MONEY:       Monetary values, including unit.
QUANTITY:    Measurements, as of weight or distance.
ORDINAL:     “first”, “second”, etc.
CARDINAL:    Numerals that do not fall under another type.
"""

# # testing

# from task_2 import load_file

# text = aggregate_text(data=load_file()[1:])
# ents = extract_named_entities(text)
# plot_entity_frequency(ents)
# raw_ents = raw_entities(text)
# damage_summary(raw_ents)
