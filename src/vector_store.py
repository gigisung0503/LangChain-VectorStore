import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def text_to_vectors(chunks):
    """Convert text chunks to embeddings."""
    vectors = [embedding_model.encode(chunk) for chunk in chunks]
    return np.array(vectors)

def build_faiss_index(vectors):
    """Build a FAISS index."""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

def query_vector_store(query, index, k=5):
    """Query the FAISS index."""
    query_vector = embedding_model.encode(query)
    distances, indices = index.search(np.array([query_vector]), k)
    return distances, indices
