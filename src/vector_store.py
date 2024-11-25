import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.pdf_processing import extract_text_from_pdf, chunk_text

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
    """
    Add new data from a PDF file to the existing FAISS index.
    
    Args:
        new_pdf_path (str): Path to the new PDF file.
        index (faiss.IndexFlatL2): Existing FAISS index.
    
    Returns:
        faiss.IndexFlatL2: Updated FAISS index with new vectors added.
    """
    
    new_document_text = extract_text_from_pdf(new_pdf_path)   # Extract text from the PDF
    new_chunks = chunk_text(new_document_text)                # Chunk the extracted text
    new_vectors = text_to_vectors(new_chunks)                 # Convert text chunks into vectors
    index.add(new_vectors)                                    # Add the vectors to the existing index
    
    return index
