from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    model.encode("Hello, world!")
