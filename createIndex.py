import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")  # Change device to "cpu" if no GPU available

# Load embeddings and corresponding text chunks
embeddings = np.load("embeddings.npy")
with open("processed_data.json", "r", encoding="utf-8") as f:
    chunked_text = json.load(f)

# Create a FAISS index
dim = embeddings.shape[1]  # Dimensionality of your embeddings (e.g., 384 for all-MiniLM-L6-v2)
index = faiss.IndexFlatL2(dim)  # L2 similarity
index.add(embeddings)  # Add embeddings to the index
faiss.write_index(index, "knowledge_base.index")  # Save the FAISS index locally

print(f"FAISS index created with {index.ntotal} entries.")

# Sample Query: Find the most relevant text chunk(s)
query = "What are the base attributes of the character?"
query_embedding = model.encode([query])  # Generate the embedding of the query

# Search FAISS index for the k closest embeddings
k = 3  # Number of results to return
distances, indices = index.search(query_embedding, k)

# Print the most relevant text chunks
print("Most relevant text chunks:")
for idx, distance in zip(indices[0], distances[0]):
    print(f"Text Chunk: {chunked_text[idx]['text']}")
    print(f"Distance: {distance}")