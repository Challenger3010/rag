import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:

    movies = load_movies()
    res = []


    for movie in movies:
        processed_query = process_text(query)
        processed_title = process_text(movie["title"])

        if processed_query in processed_title:
            res.append(movie)
            if len(res) >= limit:
                break
    return res


def process_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text