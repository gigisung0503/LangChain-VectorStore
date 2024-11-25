# Import functions from pdf_processing
from .pdf_processing import extract_text_from_pdf, chunk_text

# Import functions from vector_store
from .vector_store import (
    text_to_vectors,
    build_faiss_index,
    query_vector_store
)

# Import utility functions
from .utils import save_faiss_index, load_faiss_index
