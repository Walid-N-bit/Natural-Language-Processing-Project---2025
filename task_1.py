from newspaper import Article
import time
import newspaper
import csv
from colored_text import bcolors
import os

DEFAULT_DATA_PATH = "data/articles.csv"
DEFAULT_FIELDS = ["title", "date", "source", "article_text"]


def is_relevant(query_words: list[str], keywords: list[str]):
    qw = [w.lower() for w in query_words]
    kw = [w.lower() for w in keywords]
    return set(qw).issubset(set(kw))


def is_article_url(url: str):
    unwanted_terms = set(
        [
            "video",
            "videos",
            "live",
            "politics",
            "business",
            "ebusiness",
            "travel",
            "style",
            "culture",
            "audio",
            "subscription",
            "cnn-underscored",
            "deals",
            "podcast",
            "podcasts",
            "sport",
            "sports",
            "shop",
            "gallery",
            "pictures",
            "tech",
            "technology",
            "opinions",
            "entertainment",
            "bleacherreport.com",
            "cnnespanol.cnn.com",
            "arabic.cnn.com",
            "fi",
            "fr",
            "zh",
            "ar",
            "de",
            "es",
        ]
    )
    url_terms = url.split("/")
    for term in unwanted_terms:
        if term in url_terms:
            return False
    else:
        return True


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
        source = source.strip()
        try:
            paper = newspaper.build(source, memoize_articles=False, language="en")
            filtered_urls = [url for url in paper.article_urls() if is_article_url(url)]
            article_urls.extend(filtered_urls)
            print(
                f"{bcolors.GREEN}{len(filtered_urls)} Articles from{bcolors.ENDC} {source} {bcolors.GREEN}extracted.{bcolors.ENDC}"
            )
        except Exception as e:
            print(
                f"{bcolors.RED}Failed on {bcolors.ENDC}{source}: {bcolors.RED}{e}{bcolors.ENDC}"
            )
    return article_urls


def get_articles(
    urls: list[str], query_words: list[str], live_save: bool = False, limit: int = 30
):
    articles = []
    count = 0
    for url in urls:
        if count >= limit:
            break
        try:
            result = Article(url, language="en")
            result.download()
            result.parse()
            result.nlp()
            k_words = result.keywords
            print(f"{bcolors.BLUE}Prcessing {bcolors.ENDC}{result.url}...")
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

                print(
                    f"{bcolors.GREEN}Added {bcolors.ENDC}{result.url} {bcolors.GREEN}to articles{bcolors.ENDC}"
                )
            time.sleep(1)
        except Exception as e:
            print(
                f"{bcolors.RED}Failed on {bcolors.ENDC}{result.url}: {bcolors.RED}{e}{bcolors.ENDC}"
            )

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
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, mode="a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(DEFAULT_FIELDS)
        csvwriter.writerow(article)


# # testing
# urls = get_urls()
# qw = ["hurricane", "melissa"]
# # articles2csv()
# get_articles(urls, qw, limit=15, live_save=True)
