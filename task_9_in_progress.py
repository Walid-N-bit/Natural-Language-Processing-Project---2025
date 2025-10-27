import nltk
# nltk.download("sentiwordnet")
# nltk.download("wordnet")
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn


synsets = wn.synsets("happy", pos=wn.ADJ)
if synsets:
    happy_syn = synsets[0]
    senti = swn.senti_synset(happy_syn.name())
    print(f"Positive: {senti.pos_score()}")
    print(f"Negative: {senti.neg_score()}")
    print(f"Objective: {senti.obj_score()}")

# sentiment polarity

# emotion intensity

# damage-related words frequency
