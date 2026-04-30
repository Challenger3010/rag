import math
import os
import pickle
from utils.keyword_search_utils import tokenize_text
from utils.search_utils import load_movies, CACHE_DIR, BM25_K1, BM25_B

from collections import defaultdict, Counter

class InvertedIndex:

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_freq: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.freq_path = os.path.join(CACHE_DIR, "freq.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)

            if token not in self.term_freq:
                self.term_freq[token] = Counter()
            self.term_freq[token][doc_id] += 1
    

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
        try: 
            with open(self.docmap_path,"wb",) as f:
                pickle.dump(self.docmap, f)
            with open(self.index_path,"wb",) as f:
                pickle.dump(self.index, f)
            with open(self.freq_path,"wb",) as f:
                pickle.dump(self.term_freq, f)
            with open(self.doc_lengths_path,"wb",) as f:
                pickle.dump(self.doc_lengths, f)
        except Exception as e:
            raise Exception(f"Something went wrong saving the file", e)

    
    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.freq_path, "rb") as f:
                self.term_freq = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Index files not found, run 'build' first: {e.filename}")
        
    
    def get_tf(self, doc_id: int, term: str) -> int: 
        tokens = tokenize_text(term)

        if len(tokens) > 1:
            raise ValueError(f"Expected one token, got {len(tokens)}: {tokens}")
        return self.term_freq.get(tokens[0], Counter())[doc_id]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)

        if len(tokens) > 1:
            raise ValueError(f"Expected one token, got {len(tokens)}")

        token = tokens[0]

        n = len(self.docmap)
        df = len(self.index.get(token,set()))
        
        return math.log((n + 1)/(df + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)

        if len(tokens) > 1:
            raise ValueError(f"Expected one token, got {len(tokens)}")

        token = tokens[0]

        n = len(self.docmap)
        df = len(self.index.get(token,set()))

        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

        return idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b = BM25_B) -> float:

        avg_doc_len = self.__get_avg_doc_length()

        length_norm = (1 - b) + (b * (self.doc_lengths[doc_id] / avg_doc_len))


        raw_tf = self.get_tf(doc_id, term)
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + (k1 * length_norm))

        return bm25_tf
    
    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        res = 0.0
        
        for docs in self.doc_lengths.values():
            res += docs
        
        return (res / len(self.doc_lengths))



