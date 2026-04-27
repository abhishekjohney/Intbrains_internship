import os
import uuid

import docx
import fitz
import ollama
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


# Database setup
client = QdrantClient("http://localhost:6333")
collection_name = "universal_knowledge"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"Created new master database: {collection_name}")


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        elif ext == ".pdf":
            pdf = fitz.open(file_path)
            for page in pdf:
                text += page.get_text("text") + " "

        elif ext == ".docx":
            document = docx.Document(file_path)
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)

        elif ext in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(file_path)
            text = dataframe.to_string(index=False)

        else:
            print("Unsupported file type. Use .txt, .pdf, .docx, .xlsx, or .xls")

    except Exception as exc:
        print(f"Error reading file: {exc}")

    return text.strip()


def ingest_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        print("File not found.")
        return

    full_text = extract_text(file_path)
    if not full_text:
        print("No text found to ingest.")
        return

    words = full_text.split()
    chunk_size = 150
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

    points = []
    for chunk in chunks:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=response["embedding"],
                payload={
                    "text": chunk,
                    "source": os.path.basename(file_path),
                    "source_path": os.path.abspath(file_path),
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"Ingested {len(chunks)} chunks from {os.path.basename(file_path)}")


def list_indexed_sources() -> list[str]:
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    sources = sorted({(point.payload or {}).get("source") for point in points if (point.payload or {}).get("source")})

    if not sources:
        print("No indexed documents found.")
        return []

    print("Indexed documents:")
    for idx, source in enumerate(sources, start=1):
        print(f"{idx}. {source}")
    return sources


def chat_with_data(source_filter: str | None = None) -> None:
    print("Type 'exit' to return to the main menu.")
    if source_filter:
        print(f"Chat scope: {source_filter}")

    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() == "exit":
            break
        if not user_query:
            continue

        try:
            query_vector = ollama.embeddings(model=EMBED_MODEL, prompt=user_query)["embedding"]
        except Exception as exc:
            print(f"Embedding model error ({EMBED_MODEL}): {exc}")
            continue

        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=4,
            query_filter=query_filter,
        ).points

        if not search_results:
            print("No relevant chunks found for this scope. Upload the document first or choose a different scope.")
            continue

        context = ""
        for result in search_results:
            source = result.payload.get("source", "Unknown File")
            context += f"[Source: {source}] - {result.payload.get('text', '')}\n\n"

        system_prompt = (
            "You are a helpful assistant. Answer the user's question using only the context "
            f"provided below.\n\nCONTEXT:\n{context}"
        )

        try:
            response = ollama.chat(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
            )
            print(f"\n{CHAT_MODEL}:\n{response['message']['content']}")
        except Exception as exc:
            print(f"Chat model error ({CHAT_MODEL}): {exc}")
            print("Try: restart Ollama app/service, then run 'ollama run llama3' once in terminal.")


def main_menu() -> None:
    while True:
        print("\n1. Upload a Document")
        print("2. Chat with All Documents")
        print("3. Chat with One Document")
        print("4. Exit")

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            filepath = input("Enter the exact file path: ").strip()
            ingest_file(filepath)
        elif choice == "2":
            chat_with_data()
        elif choice == "3":
            sources = list_indexed_sources()
            if not sources:
                continue

            selected = input("Select a document number: ").strip()
            if not selected.isdigit():
                print("Please enter a valid number.")
                continue

            index = int(selected) - 1
            if index < 0 or index >= len(sources):
                print("Selection out of range.")
                continue

            chat_with_data(source_filter=sources[index])
        elif choice == "4":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")


if __name__ == "__main__":
    main_menu()
