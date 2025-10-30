import pandas as pd
import spacy
from textstat import flesch_reading_ease, gunning_fog
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

nltk.download('punkt')

# Load Dataframe
df = pd.read_csv(r'.\data\articles.csv')#We can change which DF - Its for testing

print(f"Total articles loaded: {len(df)}")

# Verify columns
if 'article_text' not in df.columns:
    raise ValueError("'article_text' column not found!")

nlp = spacy.load("en_core_web_sm")

#Functions

def calculate_ttr(text):
    """TTR"""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    types = set(tokens)
    return (len(types) / len(tokens)) * 100 if len(tokens) > 0 else 0

def calculate_msttr(text, window_size=100):
    """MSTTR"""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]

    if len(tokens) < window_size:
        return calculate_ttr(text)

    ttrs = []
    for i in range(0, len(tokens) - window_size + 1, window_size):
        window = tokens[i:i+window_size]
        if len(window) == window_size:
            ttr = len(set(window)) / len(window)
            ttrs.append(ttr)

    return (sum(ttrs) / len(ttrs)) * 100 if ttrs else calculate_ttr(text)

def calculate_lexical_density(text):
    """Lexical density with spaCy"""
    doc = nlp(text)
    content_words = [token for token in doc
                     if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']
                     and token.is_alpha]
    total_words = [token for token in doc if token.is_alpha]

    return (len(content_words) / len(total_words)) * 100 if len(total_words) > 0 else 0

def analyze_article_metrics(row):
    original_text = row['article_text']

    try:
        # Metrics
        metrics = {
            'article_id': row.name,
            'title': row['title'][:50],  
            'ttr': calculate_ttr(original_text),
            'msttr': calculate_msttr(original_text),
            'lexical_density': calculate_lexical_density(original_text),
            'flesch_reading_ease': flesch_reading_ease(original_text),
            'gunning_fog': gunning_fog(original_text),
            'word_count': len(word_tokenize(original_text)),
            'sentence_count': len(sent_tokenize(original_text)),
            'avg_sentence_length': len(word_tokenize(original_text)) / len(sent_tokenize(original_text))
        }

        return metrics

    except Exception as e:
        print(f"Error processing article {row.name}: {e}")
        return None

##Analysis

print("\nCalculating lexical metrics...")

lexical_results = []
for idx, row in df.iterrows():
    result = analyze_article_metrics(row)
    if result is not None:
        lexical_results.append(result)

lexical_df = pd.DataFrame(lexical_results)

##Results

print("\n" + "="*70)
print("LEXICAL METRICS SUMMARY")
print("="*70)
print(lexical_df.describe()) ## Summary Table

# Verificar sentence_count
print("\nSENTENCE COUNT:")
print(f"  Min sentences: {lexical_df['sentence_count'].min()}")
print(f"  Max sentences: {lexical_df['sentence_count'].max()}")
print(f"  Mean sentences: {lexical_df['sentence_count'].mean():.1f}")

#Save Results

lexical_df.to_csv(r'.\data\Lexical_Analysis.csv', index=False)
print(f"\n Results saved to 'Lexical_Analysis.csv'")

