from qdrant_client import QdrantClient
import ollama

# 1. Connect to your database
client = QdrantClient("http://localhost:6333")
collection_name = "pdf_research"

print("🤖 AI Research Assistant is online! Type 'exit' to quit.")
print("-" * 50)

# 2. The Endless Chat Loop
while True:
    user_query = input("\nYou: ")
    if user_query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    print("🔍 Searching the textbook...")

    # 3. Convert the user's question into a vector
    query_response = ollama.embeddings(
        model="nomic-embed-text", 
        prompt=user_query
    )
    query_vector = query_response["embedding"]

    # 4. Search Qdrant for the closest paragraphs
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3  # Grab the top 3 most relevant chunks
    )

    # 5. Extract the text from the database results
    retrieved_facts = ""
    for i, result in enumerate(search_results):
        # We append the text we saved earlier back together
        retrieved_facts += f"[Fact {i+1}]: {result.payload['text']}\n\n"

    # 6. Build the strict instructions for Llama 3
    system_prompt = f"""You are an expert academic assistant. 
    Answer the user's question using ONLY the provided TEXTBOOK CONTEXT. 
    If the answer is not in the context, do not guess. Simply say "I don't know based on the textbook."
    
    TEXTBOOK CONTEXT:
    {retrieved_facts}
    """

    print("🧠 Llama 3 is reading the facts and thinking...")

    # 7. Hand it all to the AI!
    response = ollama.chat(
        model="llama3", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    # Print the final answer
    print(f"\n🤖 Llama 3:\n{response['message']['content']}")