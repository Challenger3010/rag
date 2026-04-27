import string
import io
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, PROJECT_ROOT


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    query_tokens = tokenize_text(query)

    print(query_tokens)

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


def get_stop_words() -> list[str]:
    with open(f"{PROJECT_ROOT}/data/stopwords.txt","r") as f:
        content = f.read()
        row = content.splitlines()
    
    return row

def remove_stop_words(words: list[str]) -> list[str]:
    stop_words = set(get_stop_words())
    words = set(words)

    diff = words.difference(stop_words)

    return list(diff)







