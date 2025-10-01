from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load your processed text chunks
with open("processed_data.json", "r", encoding="utf-8") as f:
    chunked_text = json.load(f)

# Extract only the text chunks from your structured JSON
text_chunks = [chunk["text"] for chunk in chunked_text]

# Use a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")  # Use GPUs for faster processing

# Generate embeddings
embeddings = model.encode(text_chunks, show_progress_bar=True)

# Save embeddings for use in retrieval
np.save("embeddings.npy", embeddings)

print(f"Number of embeddings: {len(embeddings)}")  # Number of entries in embeddings.npy loaded
print(f"Number of text chunks: {len(chunked_text)}") 

print("Embeddings generated and saved as 'embeddings.npy'.")