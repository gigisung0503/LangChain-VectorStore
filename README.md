# LangChain-VectorStore

## Overview
This repository demonstrates how to process PDFs, create a vector store using FAISS, and query the store for semantic similarity tasks.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Process PDFs:
   Use the provided notebooks or `src/` scripts to extract text, chunk it, and vectorize.

3. Query the Vector Store:
   Run queries against the FAISS index.

## Repository Structure
- `notebooks/`: Jupyter Notebooks for interactive exploration.
- `src/`: Python scripts for reusable functionality.
- `data/`: Example PDFs and saved indices.

## Example
```python
from src.pdf_processing import extract_text_from_pdf, chunk_text
from src.vector_store import text_to_vectors, build_faiss_index, query_vector_store

# Load and process PDF
text = extract_text_from_pdf("data/sample.pdf")
chunks = chunk_text(text)
vectors = text_to_vectors(chunks)

# Build and query FAISS index
index = build_faiss_index(vectors)
distances, indices = query_vector_store("example query", index)
```


---

### **Benefits of This Setup**
1. **Reusability:** Modular scripts in `src/` make it easy to reuse functions.
2. **Interactive Testing:** Notebooks in `notebooks/` allow you to experiment and visualize results.
3. **Maintainability:** A structured repository simplifies collaboration and version control.
4. **Portability:** The `requirements.txt` ensures all dependencies are easily installed. 
