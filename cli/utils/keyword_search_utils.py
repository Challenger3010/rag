import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, get_stop_words
from nltk.stem import PorterStemmer


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    query_tokens = tokenize_text(query)

    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    valid_tokens = remove_stop_words(valid_tokens)
    return valid_tokens


def remove_stop_words(words: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stop_words = set(get_stop_words())
    words = set(words)

    diff = words.difference(stop_words)

    stemmed = []
    for token in list(diff):
        stemmed.append(stemmer.stem(token))

    return stemmed