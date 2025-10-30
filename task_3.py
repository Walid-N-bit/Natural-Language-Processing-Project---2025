import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def filter_english_articles_with_descriptive_stats(df, output_path="data/filtered_articles.csv"):
    """Filter non-english and duplicate articles."""

    def is_english(text):
        try:
            return detect(text) == "en"
        except:
            return False

    df["is_english"] = [is_english(text) for text in df["article_text"]]
    df_english = df[df["is_english"] == True].copy()

    # Remove duplicates and missing values
    df_english.drop_duplicates(subset=["title", "article_text"], inplace=True)
    df_english.dropna(subset=["title", "article_text"], inplace=True)

    # Summary
    total_valid_articles = len(df_english)
    avg_length = df_english["article_text"].apply(lambda x: len(x.split())).mean()
    print("\nDATA QUALITY REPORT")
    print(f"Total valid English articles: {total_valid_articles}")
    print(f"Average text length (in words): {avg_length:.2f} words")
    print(f"Number of articles before: {len(df)}")
    print(f"Number of articles now: {len(df_english)}")
    print(f"Duplicate articles removed: {len(df) - len(df_english)}")

    # Save
    df_english.to_csv(output_path, index=False)
    print(f"Saved filtered dataset to {output_path}")

# Testing
# data = pd.read_csv("data/cleaned_data.csv")
# filter_english_articles_with_descriptive_stats(data)