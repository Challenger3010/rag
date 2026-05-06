import json
import os

import numpy as np

from utils.search_utils import CACHE_DIR,semantic_chunk

from utils.semantic_search import SemanticSearch
from utils.semantic_search import cosine_similarity

class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_meta = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")

            if not text.strip():
                continue

            chunks  = semantic_chunk(text, 4, 1)

            for i,chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_meta.append({
                    "movie_idx": idx, 
                    "chunk_idx": i,
                    "total_chunks": len(chunks)
                    })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_meta

        np.save(f"{CACHE_DIR}/chunk_embeddings.npy", self.chunk_embeddings)

        try: 
            with open(f"{CACHE_DIR}/chunk_metadata.json", "w") as f:
                json.dump({"chunks": chunk_meta, "total_chunks": len(all_chunks)}, f, indent=2)
        except Exception as e:
            raise Exception(f"Something went wrong saving the file", e)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc
    

        path = f"{CACHE_DIR}/chunk_embeddings.npy"
        json_path = f"{CACHE_DIR}/chunk_metadata.json"

        if os.path.exists(path) and os.path.exists(json_path):
            self.chunk_embeddings = np.load(path)
            with open(json_path, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
    
    def search_chunk(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embedding loaded.")

        query = query.strip()


        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            sim = cosine_similarity(query_embedding, chunk_embedding)
            meta = self.chunk_metadata[i]

            chunk_scores.append({
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": sim 
                })
            
            movie_scores = {}

            for chunk_score in chunk_scores:
                movie_idx = chunk_score["movie_idx"]
                if movie_idx not in movie_scores or chunk_score["score"] > movie_scores[movie_idx]:
                   movie_scores[movie_idx] = chunk_score["score"]
            
            sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

            results = []
            for movie_idx, score in sorted_movies[:limit]:
                if movie_idx is None:
                    continue

                doc = self.documents[movie_idx]

                results.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score, 4),
                    })
    
        return results









        







