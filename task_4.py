import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

def plot_and_save(freq_data, title_prefix, score_label, file_prefix):
    """To create and save bar chart and word cloud."""
    # Bar Chart
    words, scores = zip(*freq_data[:20])
    plt.figure(figsize=(10, 5))
    plt.barh(words[::-1], scores[::-1])
    plt.title(f"Top 20 {title_prefix} Words")
    plt.xlabel(score_label)
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(f"top20_{file_prefix}_bar.png", dpi=300)
    plt.show()

    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies(dict(freq_data))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {title_prefix}")
    plt.tight_layout()
    plt.savefig(f"wordcloud_{file_prefix}.png", dpi=300)
    plt.show()

def keyword_analysis(df, output_path="data/articles_top_keywords.csv"):
    """
    Extract top keywords using both CountVectorizer and TF-IDF,
    visualize results in bar charts and word clouds
    """

    # CountVectorizer
    cv = CountVectorizer(max_features=2000)
    word_count = cv.fit_transform(df['clean_text'])
    sum_words = word_count.sum(axis=0)
    words_freq = sorted([(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()],
                        key=lambda x: x[1], reverse=True)
    plot_and_save(words_freq, "CountVectorizer (Frequency)", "Count", "countvectorizer")

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])
    tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    tfidf_words = tfidf.get_feature_names_out()
    tfidf_freq = sorted(list(zip(tfidf_words, tfidf_scores)), key=lambda x: x[1], reverse=True)
    plot_and_save(tfidf_freq, "TF-IDF (Importance)", "TF-IDF Score", "tfidf")

    # Top TF-IDF keywords per article
    def extract_top_tfidf(row, feature_names, matrix, top_n=5):
        row_data = matrix[row].toarray().flatten()
        top_indices = row_data.argsort()[-top_n:][::-1]
        return ", ".join([feature_names[i] for i in top_indices if row_data[i] > 0])

    df['top_keywords'] = [extract_top_tfidf(i, tfidf_words, tfidf_matrix)
                          for i in range(tfidf_matrix.shape[0])]

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n Keyword extraction complete! Saved to {output_path}")

# Testing
# data = pd.read_csv("data/filtered_articles.csv")
# keyword_analysis(data)