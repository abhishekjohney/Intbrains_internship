import fitz  # This is the PyMuPDF library we just installed!
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
import uuid

# 1. Connect to Qdrant and set up the collection
client = QdrantClient("http://localhost:6333")
collection_name = "pdf_research"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"📦 Created new database table: {collection_name}")

# 2. Open and Read the PDF
pdf_path = "textbook.pdf" 
print(f"📄 Cracking open {pdf_path}...")

doc = fitz.open(pdf_path)
full_text = ""

# Extract text from every single page
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    full_text += page.get_text("text") + " "

print("✅ PDF text extracted successfully.")

# 3. The "Chunking" Engine
# We split the massive wall of text into smaller 150-word blocks
words = full_text.split()
chunk_size = 150
chunks = []

for i in range(0, len(words), chunk_size):
    chunk_string = " ".join(words[i : i + chunk_size])
    chunks.append(chunk_string)

print(f"✂️ Chopped the PDF into {len(chunks)} chunks.")

# 4. Convert to Vectors and Upload
points = []
print("🧠 Embedding chunks and uploading to Docker Qdrant...")

for i, chunk in enumerate(chunks):
    # Ask Ollama to turn the chunk into 768 dimensions
    response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
    
    # Package it for Qdrant (Notice the payload!)
    points.append(
        PointStruct(
            id=str(uuid.uuid4()), 
            vector=response["embedding"],
            payload={"text": chunk, "chunk_index": i, "source": pdf_path} 
        )
    )

client.upsert(collection_name=collection_name, points=points)
print("\n🚀 SUCCESS! The PDF is now permanently memorized in Qdrant.")