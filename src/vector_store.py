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

def add_new_data_to_index(new_pdf_path, index):
    """Add new pdf."""
    new_document_text = extract_text_from_pdf(new_pdf_path)  # Extract text from the new PDF
    new_chunks = chunk_text(new_document_text)               # Chunk the new document
    new_vectors = text_to_vectors(new_chunks)                # Convert new chunks to vectors
    index.add(new_vectors)                                   # Add new vectors to the existing index
    return index
