"""
dashboard should have:
- emotion score and
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
import seaborn as sns


def zipf_graph(df: pd.DataFrame):
    """Perform Zipf’s Law analysis and visualize rank-frequency relationship."""

    texts_df = df["clean_text"]
    texts = list(texts_df)

    # corpus data
    corpus = " ".join(texts)

    def process_text(text: str):
        tokens = word_tokenize(text)
        freq_dist = Counter(tokens)
        ranks = list(range(1, len(freq_dist) + 1))
        total = sum(freq_dist.values())
        ranks = list(range(1, len(freq_dist) + 1))
        frequencies = [freq / total for _, freq in freq_dist.most_common()]
        return ranks, frequencies

    corpus_ranks, corpus_freqs = process_text(corpus)

    # ideal Zipf's law data
    zipf_freqs = [corpus_freqs[0] / rank for rank in corpus_ranks]

    # per article data
    texts_ranks_freqs = []
    for row in texts:
        ranks, freqs = process_text(row)
        texts_ranks_freqs.append([ranks, freqs])

    lines = texts_ranks_freqs
    num_lines = st.sidebar.slider(
        "Number of articles to show",
        min_value=1,
        max_value=len(texts_ranks_freqs),
        value=2,
    )

    fig, ax = plt.subplots()
    plt.loglog(
        corpus_ranks, corpus_freqs, color="#026400FF", label="Corpus of all articles"
    )
    plt.loglog(
        corpus_ranks,
        zipf_freqs,
        linestyle="--",
        color="#FF33336C",
        label="Ideal Zipf's law",
    )
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Zipf's Law: Rank vs Frequency")

    for i in range(num_lines):
        ax.loglog(
            lines[i][0],
            lines[i][1],
            color="#3347FF6A",
            linestyle="-",
            alpha=0.4,
            label="articles",
        )

    return fig


def emotion_heatmap(data: pd.DataFrame):
    fig, ax = plt.subplots()
    plt.xticks(rotation=45)
    plt.yticks(rotation=90)
    plt.tight_layout()

    sns.heatmap(data, annot=True, cmap="YlOrRd", ax=ax)
    return fig


# Main dashboard function
def streamlit_dash(clean_data: pd.DataFrame, emotion_data: pd.DataFrame):

    sidebar_tab = st.sidebar.radio(
        "Navigation", ["Zipf Graph", "Emotions intensity Heatmap", "More"], index=0
    )

    # Display based on sidebar tab
    if sidebar_tab == "Zipf Graph":
        st.header("Zipf's Law Visualization")
        fig = zipf_graph(clean_data)
        st.pyplot(fig)

    elif sidebar_tab == "Emotions intensity Heatmap":
        st.header("Other Analysis")
        st.info("This section is under construction.")
        fig = emotion_heatmap(emotion_data)
        st.pyplot(fig)

    elif sidebar_tab == "More":
        st.header("More Visualizations")
        st.info("Future sections will appear here.")


# Testing

data1 = pd.read_csv("data/cleaned_data.csv")
data2 = pd.read_csv(
    "data/sentiment_emotion_analysis.csv",
    usecols=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
)
streamlit_dash(data1, data2)
