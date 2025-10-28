import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tag import pos_tag

wnl = nltk.WordNetLemmatizer()

synsets = wn.synsets("death", pos=wn.NOUN)
print(pos_tag(["death"]))
if synsets:
    happy_syn = synsets[0]
    senti = swn.senti_synset(happy_syn.name())
    print(f"Positive: {senti.pos_score()}")
    print(f"Negative: {senti.neg_score()}")
    print(f"Objective: {senti.obj_score()}")

# sentiment polarity





def sentiment_polarity(lemmatized_tokens: list[str]):
    pos_sentiment = 0
    neg_sentiment = 0


# emotion intensity

# damage-related words frequency

print(
    [
        wnl.lemmatize(w.lower(), pos=wn.VERB)
        for w in "Melissa rapidly intensified over the weekend and is now a rare Category 5 hurricane with 160 mph winds.".split()
    ]
)
