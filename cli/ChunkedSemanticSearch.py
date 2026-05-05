import json
import os

import numpy as np

from utils.search_utils import CACHE_DIR,semantic_chunk

from utils.semantic_search import SemanticSearch

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        chunks = []
        chunk_meta = []

        for doc_idx ,doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
            if len(doc["description"]) == 0:
                continue

            res  = semantic_chunk(doc["description"], 4, 1)

            for i,chunk in enumerate(res):
                chunks.append(chunk)
                chunk_meta.append({
                    "movie_idx": doc["id"],
                    "chunk_idx": i,
                    "total_chunks": len(res)
                    })

        print(f"Total chunks to encode: {len(chunks)}")
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_meta

        np.save(f"{CACHE_DIR}/chunk_embeddings.npy", self.chunk_embeddings)

        try: 
            with open(f"{CACHE_DIR}/chunk_metadata.json", "w") as f:
                json.dump({"chunks": chunk_meta, "total_chunks": len(chunks)}, f, indent=2)
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
        







