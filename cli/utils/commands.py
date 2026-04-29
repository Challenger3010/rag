from InvertedIndex import InvertedIndex
from .keyword_search_utils import tokenize_text
from .search_utils import DEFAULT_SEARCH_LIMIT

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


def build_command():
    idx = InvertedIndex()
    print(f"Building the inverted index...")
    idx.build()
    print(f"Saving the index...")
    idx.save()
    print("index saved succefully")


def tf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()
    freq = idx.get_tf(doc_id, term)

    print("Frequency")
    print(freq)

def idf_command(term: str):
    idx = InvertedIndex()
    idx.load()
    print(f"Inverse document frequency of '{term}': {idx.get_idf(term):.2f}")