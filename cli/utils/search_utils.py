import json
import os
import re
from nltk.stem import PorterStemmer


DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
BM25_K1 = 1.5
BM25_B = 0.75


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def get_stop_words() -> list[str]:
    with open(f"{STOPWORDS_PATH}","r") as f:
        content = f.read()
        row = content.splitlines()
    
    return row

def remove_stop_words(words: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stop_words = set(get_stop_words())
    stemmed = []

    for token in words:
        if token not in stop_words:
            stemmed.append(stemmer.stem(token))
    
    return stemmed

def semantic_chunk(text: str, max_chunk_size = 4, overlap = 0):
    counter = 0
    for c in text:
        counter += 1

    splitted_text = re.split(r"(?<=[.!?])\s+", text)
    i = 0
    res = []

    while i < len(splitted_text):
        chunk = splitted_text[i: max_chunk_size + i]
        res.append(" ".join(chunk))
        i += max_chunk_size - overlap

    # print(f"Semantically chunking {counter} characters")
    # for j,words in enumerate(res,1):
    #     print(f"{j}. {words}")
    
    return res