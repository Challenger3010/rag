from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:

    movies = load_movies()
    res = []

    for movie in movies:
        if query in movie["title"]:
            res.append(movie)
            if len(res) >= limit:
                break
    return res