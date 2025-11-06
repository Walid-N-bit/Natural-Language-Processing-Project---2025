from newspaper import Article
import time
import newspaper
import csv
from colored_text import bcolors
import os
from datetime import datetime
import re

DEFAULT_DATA_PATH = "data/articles.csv"
DEFAULT_FIELDS = ["title", "date", "source", "article_text"]


def is_relevant(query_words: list[str], keywords: list[str]):
    """
    compare keywords in qwery with keywords in article
    """
    qw = [w.lower() for w in query_words]
    kw = [w.lower() for w in keywords]
    return set(qw).issubset(set(kw))


# def is_article_url(url: str):
#     """
#     compare link path terms with non-article related terms, return true if the link doesn't
#     contain any of them. return false otherwise
#     """
#     unwanted_terms = set(
#         [
#             "video",
#             "videos",
#             "live",
#             "politics",
#             "business",
#             "ebusiness",
#             "travel",
#             "style",
#             "culture",
#             "audio",
#             "subscription",
#             "cnn-underscored",
#             "deals",
#             "podcast",
#             "podcasts",
#             "sport",
#             "sports",
#             "shop",
#             "gallery",
#             "pictures",
#             "tech",
#             "technology",
#             "opinions",
#             "entertainment",
#             "bleacherreport.com",
#             "cnnespanol.cnn.com",
#             "arabic.cnn.com",
#             "fi",
#             "fr",
#             "zh",
#             "ar",
#             "de",
#             "es",
#         ]
#     )
#     url_terms = url.split("/")
#     for term in unwanted_terms:
#         if term in url_terms:
#             return False
#     else:
#         return True


def is_article_url(url: str):
    """
    Return True if the URL likely points to an article (i.e., it doesn't contain
    unwanted terms anywhere in domain or path). Return False otherwise.
    """
    unwanted_terms = [
        "video",
        "videos",
        "live",
        "politics",
        "business",
        "finance",
        "ebusiness",
        "travel",
        "style",
        "fashion",
        "beauty",
        "culture",
        "lifestyle",
        "health",
        "science",
        "audio",
        "subscription",
        "cnn-underscored",
        "deals",
        "podcast",
        "podcasts",
        "sport",
        "sports",
        "shop",
        "shopping",
        "local",
        "gallery",
        "pictures",
        "tech",
        "technology",
        "autos",
        "opinions",
        "entertainment",
        "bleacherreport\\.com",
        "cnnespanol\\.cnn\\.com",
        "arabic\\.cnn\\.com",
        "fi",
        "fr",
        "zh",
        "ar",
        "de",
        "es",
        "es-us",
        "tw",
        "hk",
    ]

    pattern = re.compile(
        r"(^|[./])(" + "|".join(unwanted_terms) + r")([./]|$)", re.IGNORECASE
    )

    return not bool(pattern.search(url))


def get_urls(sources=[]):
    """
    get all links from sources list. If no sources were passed in params, a default list is used
    """
    article_urls = []
    if len(sources) == 0:
        sources = [
            "https://www.yahoo.com/news/world/",
            "https://edition.cnn.com/world",
            "https://www.aljazeera.com/news",
            "https://www.bbc.com/news",
            "https://www.cbsnews.com/",
            "https://www.npr.org/sections/world/",
        ]
    for source in sources:
        source = source.strip()
        try:
            paper = newspaper.build(source, memoize_articles=False, language="en")
            filtered_urls = [url for url in paper.article_urls() if is_article_url(url)]
            article_urls.extend(filtered_urls)
            print(
                f"{len(filtered_urls)} {bcolors.GREEN}Articles from{bcolors.ENDC} {source} {bcolors.GREEN}extracted.{bcolors.ENDC}"
            )
        except Exception as e:
            print(
                f"{bcolors.RED}Failed on {bcolors.ENDC}{source}: {bcolors.RED}{e}{bcolors.ENDC}"
            )
    # save articles urls to a csv file
    path = "data/articles_urls.csv"
    fields = ["url"]
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
    existing_urls = set()
    if file_exists:
        with open(path, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            existing_urls = {row[0] for row in reader if row}

    with open(path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(fields)
        for url in article_urls:
            if url not in existing_urls:
                writer.writerow([url])

    return article_urls


def get_articles(
    urls: list[str], query_words: list[str], live_save: bool = False, limit: int = 30
):
    """
    download and parse article data. If article is not relevant according to query, discard it.
    relevant article data are saved to a csv file.
    """
    begin_t = datetime.now()
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
    end_t = datetime.now()
    print(f"\nProcess finished!\nElapsed time: {end_t - begin_t}")
    return articles


def articles2csv(
    articles: list[list[str]] = [], path: str = DEFAULT_DATA_PATH, fields=DEFAULT_FIELDS
):
    """
    save list of article data to a csv file
    """
    with open(path, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        if len(articles) > 0:
            csvwriter.writerows(articles)


def append_article_to_csv(article: list[str], path: str = DEFAULT_DATA_PATH):
    """
    append data of one article to a csv file. if no file exists, one is created.
    """
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
