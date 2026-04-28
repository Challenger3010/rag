import os
import pickle
from utils.keyword_search_utils import tokenize_text
from utils.search_utils import load_movies, CACHE_DIR

from collections import defaultdict

class InvertedIndex:

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
    

    def get_document(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        return sorted(list(doc_ids))

    
    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            doc_desc = f"{movie["title"]} {movie["description"]}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_desc)




    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.docmap_path,"wb",) as f:
            pickle.dump(self.docmap, f)
        with open(self.index_path,"wb",) as f:
            pickle.dump(self.index, f)