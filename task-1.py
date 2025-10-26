from newspaper import Article
import time
import newspaper
import csv

DEFAULT_DATA_PATH = "data/articles.csv"
DEFAULT_FIELDS = ["title", "date", "source", "article_text"]


def is_relevant(query_words: list[str], keywords: list[str]):
    qw = [w.lower() for w in query_words]
    kw = [w.lower() for w in keywords]
    return set(qw).issubset(set(kw))


def get_urls(sources=[]):
    article_urls = []
    if len(sources) == 0:
        sources = [
            "https://edition.cnn.com/world",
            "https://www.aljazeera.com/news",
            "https://www.bbc.com/news",
            "https://www.cbsnews.com/",
        ]
    for source in sources:
        paper = newspaper.build(source, memoize_articles=False, language="en")
        for url in paper.article_urls():
            article_urls.append(url)
        print(f"{len(paper.article_urls())} Articles from {source} extracted.")
    return article_urls


def get_articles(
    urls: list[str], query_words: list[str], live_save: bool = False, limit: int = 30
):
    articles = []
    i = 0
    for url in urls:
        result = Article(url, language="en")
        result.download()
        result.parse()
        result.nlp()
        k_words = result.keywords
        print(f"Prcessing {result.url}...")
        if is_relevant(query_words, k_words):
            article = [
                result.title,
                str(result.publish_date),
                result.source_url,
                result.text,
            ]
            articles.append(article)
            if live_save:
                append_article_to_csv(article)

            print(f"Added {result.url} to articles")
        time.sleep(1)

        i += 1
        if i == limit:
            continue
    return articles


def articles2csv(
    articles: list[list[str]] = [], path: str = DEFAULT_DATA_PATH, fields=DEFAULT_FIELDS
):
    with open(path, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        if len(articles) > 0:
            csvwriter.writerows(articles)


def append_article_to_csv(article: list[str], path: str = DEFAULT_DATA_PATH):
    with open(path, mode="a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(article)


# testing
# urls = get_urls()
# qw = ["hurricane", "melissa"]
# articles2csv()
# get_articles(urls, qw, limit=3, live_save=True)
