import json
from PyPDF2 import PdfReader
import re
import configparser
from transformers import AutoTokenizer
from pathlib import Path

#Init configuration
config = configparser.ConfigParser()
config.read('config.ini')

print(config.sections())

# Configure your model path
model_path = config.get("model","path")
source_path = config.get("books","source_path")
print(f"Using model path: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Extract and clean all page text into one large string
full_text = ""
for book in json.loads(config.get("books","sources")):
    # Define the number of pages to ignore at the start and end
    IgnoreFirstN = book["IgnoreFirstN"]  # Number of pages to ignore from the start
    IgnoreLastN = book["IgnoreLastN"]   # Number of pages to ignore from the end

    # Step 1: Read and clean the full PDF
    book_path = Path(f"{source_path}\{book['path']}")
    reader = PdfReader()

    # Get the total number of pages in the PDF
    total_pages = len(reader.pages)

    # Calculate the range of pages to process
    start_page = IgnoreFirstN
    end_page = total_pages - IgnoreLastN

    # Validate the range (ensure it's valid)
    if start_page < 0 or end_page > total_pages or start_page >= end_page:
        raise ValueError("Invalid 'IgnoreFirstN' and 'IgnoreLastN' values. Ensure the range is valid.")

    # Iterate over the valid page range
    for page_number in range(start_page, end_page):
        page = reader.pages[page_number]
        # Extract text from the page and clean it
        page_text = page.extract_text().strip() if page.extract_text() else ""
        page_text = re.sub(r"\s+", " ", page_text)  # Replace multiple spaces with a single space
        page_text = page_text.replace(book["waterMark"], "").strip()  # Remove watermark (if defined)
        full_text += page_text + " "  # Add spacing between pages

    print(f"Total length of full text: {len(full_text)} characters.")

# Step 2: Split full cleaned text into tokenized chunks
def split_into_chunks_with_tokens(text, max_tokens=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(tokens), max_tokens - overlap):  # Includes overlap for context preservation
        chunk = tokens[start:start + max_tokens]
        chunks.append({"tokens": chunk, "text": tokenizer.decode(chunk)})
    return chunks

# Generate chunks from the cleaned "full_text"
chunked_text = split_into_chunks_with_tokens(full_text)

# Step 3: Save the processed chunks
with open(".\data\processed_data.json", "w", encoding="utf-8") as f:
    json.dump(chunked_text, f, indent=4)

print(f"Total chunks created: {len(chunked_text)}")