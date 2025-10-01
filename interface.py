import os
import gradio as gr
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Log in to Hugging Face to access private models if needed
HF_TOKEN = os.getenv("HF_TOKEN")

login(token = HF_TOKEN)

# Constants:
EMBEDDINGS_FILE = "embeddings.npy"                # Generated Embeddings (NumPy file)
CHUNKED_TEXT_FILE = "processed_data.json"  # Processed Data Chunks (JSON file)
FAISS_INDEX_FILE = "knowledge_base.index"         # FAISS Index File

# model_path = "meta-llama/Llama-3.1-8B"
# model_path = "nvidia/Mistral-NeMo-12B-Instruct"
model_path = "mistralai/Mistral-Nemo-Instruct-2407"
global_context = """
You are a helpful assistant, a Rulemaster for the RPG game Forbidden Lands. Your task is to answer rule and lore questions based on the context provided from the game's rulebooks. 
Use the context to provide accurate and concise answers. 
If the context does not contain the information needed to answer a question, respond with "I don't know" or "The provided context does not cover that information." 
Always aim to assist the user in understanding the rules and lore of Forbidden Lands.
Once you answer a question, you MUST conclude your response and avoid expanding the answer
with unnecessary information or asking yourself additional questions.
Focus on directly addressing the user's query and never create follow-up questions unless the user explicitly asks for more.
"""


# Load the LLM
llm_path = model_path  # Replace with your fine-tuned or pre-trained LLM model path
llm_model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto", load_in_8bit=True)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)

# 1. Load the Sentence Transformer Model
def initialize_model():
    print("Initializing the embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")  # Use "cpu" if GPU isn't available
    return model


# 2. Load Text Chunks and Generated Embeddings
def load_data(embeddings_file, text_file):
    print(f"Loading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)  # Loading the embeddings from saved NumPy file
    print(f"Loading text chunks from {text_file}...")
    with open(text_file, "r", encoding="utf-8") as f:
        chunked_text = json.load(f)  # Loading processed JSON text chunks
    return embeddings, chunked_text


# 3. Build/Load FAISS Index
def build_or_load_faiss_index(embeddings, index_file, dim):
    if not index_file:
        print("Creating a FAISS index...")
        index = faiss.IndexFlatL2(dim)  # Using L2 (Euclidean distance) for nearest neighbor search
        index.add(embeddings)          # Add embeddings to FAISS index
        faiss.write_index(index, index_file)  # Save FAISS index to a file for future use
        print(f"FAISS index created and saved as '{index_file}'.")
    else:
        print(f"Loading FAISS index from '{index_file}'...")
        index = faiss.read_index(index_file)  # Load FAISS index from file
    return index


# 4. Perform Query to Find Relevant Text
def perform_query(query, model, index, chunked_text):
    print(f"Performing query: {query}")
    # Embed the query using the sentence-transformers model
    query_embedding = model.encode([query])
    k = 5  # Number of matching results to retrieve
    distances, indices = index.search(query_embedding, k)

    # Output the top-k most relevant results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        results.append({
            "text": chunked_text[idx]["text"] if isinstance(chunked_text[idx], dict) else chunked_text[idx],
            "distance": distance
        })
    return results


# 5. Gradio Chat Interface
def lore_master(query):
    results = perform_query(query, model, index, chunked_text)  # Perform query on the FAISS index
    top_result = results[0]  # Take the first/top match
    return f"--- Context Retrieved ---\n{top_result['text']}\n\n--- User Question ---\n{query}"

def lore_master_with_llm(query):
    results = perform_query(query, model, index, chunked_text)
    if not results:
        return "No valid chunks found for this query."
    top_result = results[0]["text"]
    # Build a prompt for the LLM
    prompt = f"{global_context}\n\n{top_result}\n\nUser question: {query}\nAnswer:"
    inputs = llm_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llm_model.generate(
        inputs.input_ids,
        max_new_tokens=150,
        temperature=0.6,              # Reduce randomness
        top_p=0.9,                    # Nucleus sampling for top 90% tokens
        top_k=50,                     # Focus on top-k likely tokens
        repetition_penalty=1.1        # Discourage repetitive answers
    )
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return f"--- Context Retrieved ---\n{top_result}\n\n--- Assistant Response ---\n{answer}"


# MAIN PROGRAM
if __name__ == "__main__":
    # Initialize embedding model
    model = initialize_model()

    # Load embeddings and text chunks
    embeddings, chunked_text = load_data(EMBEDDINGS_FILE, CHUNKED_TEXT_FILE)

    # Get embedding dimensionality
    dim = embeddings.shape[1]

    # Build or load FAISS index
    index = build_or_load_faiss_index(embeddings, FAISS_INDEX_FILE, dim)

    # Set up Gradio interface
    print("Launching Gradio interface...")
    gr.Interface(
        fn=lore_master_with_llm,
        inputs="text",
        outputs=[gr.Textbox(label="output", lines=3)],
        title="Lore Master Assistant").launch()