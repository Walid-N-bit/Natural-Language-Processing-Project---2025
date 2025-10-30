import pandas as pd
import nltk
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

def zipf_analysis(df):
    """Perform Zipf’s Law analysis and visualize rank-frequency relationship."""
    corpus = " ".join(df['clean_text'].astype(str))
    tokens = word_tokenize(corpus.lower())
    tokens = [t for t in tokens if t.isalpha()]

    freq_dist = Counter(tokens)
    ranks = range(1, len(freq_dist) + 1)
    frequencies = [freq for word, freq in freq_dist.most_common()]

    # Plot Rank vs Frequency on a log-log scale
    plt.figure(figsize=(8,6))
    plt.loglog(ranks, frequencies, marker=".")
    plt.title("Zipf’s Law: Rank vs Frequency")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.grid(True)
    plt.savefig("zipf_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

     # Fit a regression line on log-transformed data to estimate slope
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    slope, _ = np.polyfit(log_ranks, log_freqs, 1)
    print(f"Estimated Zipf slope: {slope:.2f}")
    if -1.3 < slope < -0.7:
        print("Corpus roughly follows Zipf’s Law.")
    else:
        print("Corpus deviates from Zipf’s Law.")

# Testing
# data = pd.read_csv("data/filtered_articles.csv")
# zipf_analysis(data)