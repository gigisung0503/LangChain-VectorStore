import faiss

def save_faiss_index(index, path="faiss_index.bin"):
    """Save the FAISS index to a file."""
    faiss.write_index(index, path)

def load_faiss_index(path="faiss_index.bin"):
    """Load the FAISS index from a file."""
    return faiss.read_index(path)
