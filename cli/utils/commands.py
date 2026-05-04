from InvertedIndex import InvertedIndex
from .keyword_search_utils import tokenize_text
from .search_utils import DEFAULT_SEARCH_LIMIT
from .semantic_search import SemanticSearch

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:

    results = []
    seen = set()
    query_tokens = tokenize_text(query)
    idx = InvertedIndex()
    idx.load()

    for token in query_tokens:
        ids = idx.get_document(token)
        for doc_id in ids:
            if doc_id not in seen:
                seen.add(doc_id)
                results.append(idx.docmap[doc_id])

            if len(results) >= limit:
                return results

    return results


def build_command() -> None:
    idx = InvertedIndex()
    print(f"Building the inverted index...")
    idx.build()
    print(f"Saving the index...")
    idx.save()
    print("index saved succefully")


def tf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    idx.load()
    freq = idx.get_tf(doc_id, term)

    print("Frequency")
    print(freq)

def idf_command(term: str) -> None:
    idx = InvertedIndex()
    idx.load()
    print(f"Inverse document frequency of '{term}': {idx.get_idf(term):.2f}")

def tfidf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    idx.load()

    tf = idx.get_tf(doc_id, term)
    idf = idx.get_idf(term)

    tfidf = tf * idf

    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf:.2f}")

def bm25_idf_command(term: str) -> None:
    idx = InvertedIndex()
    idx.load()

    print(f"BM25 IDF score of '{term}': {idx.get_bm25_idf(term):.2f}")

def bm25_tf_command(doc_id: int, term: str, k1, b: float) -> None:
    idx = InvertedIndex()
    idx.load()

    tf = idx.get_bm25_tf(doc_id, term, k1, b)

    print(f"BM25 TF score of '{term}' in document '{doc_id}': {tf:.2f}")

def bm25search_command(query):
    idx = InvertedIndex()
    idx.load()

    movies = idx.bm25_search(query)

    for i, (score, movie) in enumerate(movies, start=1):
        print(f"{i}. ({movie["id"]}) {movie["title"]} - {score:.2f}")

def verify_command():
    searcher = SemanticSearch()
    searcher.verify_model()

def embed_text_command(text):
    searcher = SemanticSearch()
    searcher.embed_text(text)

