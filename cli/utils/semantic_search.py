from sentence_transformers import SentenceTransformer
class SemanticSearch:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    
    def verify_model(self):
        searcher = SemanticSearch()
        print(f"Model loaded: {searcher.model}")
        print(f"Max sequence length: {searcher.model.max_seq_length}")
    
    def generate_embedding(self, text: str):
        if len(text) == 0 or text.isspace():
            raise ValueError("Text can not be empty or whitespaces")
        
        embedding = self.model.encode(list(text))

        return embedding[0]
    
    def embed_text(self, text):
        searcher = SemanticSearch()
        embedding = searcher.generate_embedding(text)

        print(f"Text: {text}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")
 


