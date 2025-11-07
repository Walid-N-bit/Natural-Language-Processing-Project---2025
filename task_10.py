import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


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


def sentiment_scatterplot(data):
    """
    correlation between emotion intensity and score impact
    """
    st.caption(
        "Scatter plot showing correlation between Emotion Intensity, Sentiment Polarity/Damage Keywords Frequency/Impact Score."
        "\nHold and drag to view."
    )
    x = data["neg_emo_intensity"]
    y = data["neu_emo_intensity"]
    z = data["pos_emo_intensity"]
    c = []
    c_metrics = st.sidebar.radio(
        "Metrics",
        ["Sentiment Polarity", "Damage Keyword Frequency", "Impact Score"],
        index=0,
    )
    if c_metrics == "Sentiment Polarity":
        c = data["sentiment_polarity"]
    elif c_metrics == "Damage Keyword Frequency":
        c = data["damage_frequency"]
    elif c_metrics == "Impact Score":
        c = data["impact_score"]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=6,
                    color=c,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title=c_metrics),
                ),
            )
        ]
    )

    fig.update_layout(
        font=dict(color="black"),
        paper_bgcolor="white",
        title="3D Sentiment Scatter Plot (Interactive)",
        scene=dict(
            xaxis_title="Negative Emotion Intensity",
            yaxis_title="Neutral Emotion Intensity",
            zaxis_title="Positive Emotion Intensity",
        ),
        height=700,
    )

    # st.plotly_chart(fig, use_container_width=True)
    return fig


# Main dashboard function
def streamlit_dash(
    clean_data: pd.DataFrame, emotion_data: pd.DataFrame, score_data: pd.DataFrame
):

    sidebar_tab = st.sidebar.radio(
        "Navigation",
        ["Zipf Graph", "Emotions intensity Heatmap", "Scatter Plot"],
        index=0,
    )

    # Display based on sidebar tab
    if sidebar_tab == "Zipf Graph":
        st.header("Zipf's Law Visualization")
        fig = zipf_graph(clean_data)
        st.pyplot(fig)

    elif sidebar_tab == "Emotions intensity Heatmap":
        st.header("Other Analysis")
        fig = emotion_heatmap(emotion_data)
        st.pyplot(fig)

    elif sidebar_tab == "Scatter Plot":
        st.header("Scatter Plot for Emotion Intensity")
        fig = sentiment_scatterplot(score_data)
        # st.pyplot(fig)
        st.plotly_chart(fig, use_container_width=True)


# Testing

data1 = pd.read_csv("data/cleaned_data.csv")
data2 = pd.read_csv(
    "data/sentiment_emotion_analysis.csv",
    usecols=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
)
data3 = pd.read_csv(
    "data/scored_articles.csv",
    usecols=[
        "neg_emo_intensity",
        "neu_emo_intensity",
        "pos_emo_intensity",
        "sentiment_polarity",
        "damage_frequency",
        "impact_score",
    ],
)

streamlit_dash(data1, data2, data3)
