from sentence_transformers import SentenceTransformer
from .search_utils import CACHE_DIR, load_movies

import os
import numpy as np

class SemanticSearch:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def verify_model(self) -> None:
        searcher = SemanticSearch()
        print(f"Model loaded: {searcher.model}")
        print(f"Max sequence length: {searcher.model.max_seq_length}")
    
    def verify_embeddings(self):
        searcher = SemanticSearch()
        documents = load_movies()

        embeddings = searcher.load_or_create_embedings(documents)

        print(f"Number of docs:   {len(documents)}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

    
    def generate_embedding(self, text: str) -> list[str]:
        if len(text) == 0 or text.isspace():
            raise ValueError("Text can not be empty or whitespaces")
        
        embedding = self.model.encode([text])

        return embedding[0]
    
    def embed_text(self, text: str) -> None:
        searcher = SemanticSearch()
        embedding = searcher.generate_embedding(text)

        print(f"Text: {text}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")
 
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}

        doc_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc

            doc_list.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)

        np.save(f"{CACHE_DIR}/movie_embeddings.npy", self.embeddings)

        return self.embeddings
    
    def load_or_create_embedings(self, documents):
        self.documents = documents
        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        path = f"{CACHE_DIR}/movie_embeddings.npy"
        if os.path.exists(path):
            self.embeddings = np.load(path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)
    
    def embed_query_text(self, query):
        searcher = SemanticSearch()
        embedding = searcher.generate_embedding(query)

        print(f"Query: {query}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Shape: {embedding.shape}")
    
    def search(self, query, limit):

        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        
        embdeding = self.generate_embedding(query)

        sim_list = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = cosine_similarity(embdeding, doc_emb)

            if sim > 0.0:
                sim_list.append((sim, self.documents[i]))
        
        sim_list.sort(key=lambda x: x[0], reverse=True)
        
        return [{"score": s, "title": doc["title"], "description": doc["description"]} for s, doc in sim_list[:limit]]



def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

        


        



        




