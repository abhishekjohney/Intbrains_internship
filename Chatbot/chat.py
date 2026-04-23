import urllib.request
import json

# The URL for your local Ollama engine
OLLAMA_URL = "http://localhost:11434/api/chat"

# This list will hold the memory of our conversation
chat_history = []

print("=== Local Llama 3 Terminal Chat ===")
print("Type 'quit' or 'exit' to stop the program.\n")

while True:
    # 1. Get input from the terminal
    user_text = input("You: ")
    
    # Check if the user wants to leave
    if user_text.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break
        
    if not user_text.strip():
        continue

    # 2. Add the user's message to the history
    chat_history.append({"role": "user", "content": user_text})

    # 3. Format the data exactly how Ollama expects it
    payload = {
        "model": "llama3",
        "messages": chat_history,
        "stream": False
    }

    # Convert the Python dictionary into a JSON string, then encode it into raw bytes
    data_bytes = json.dumps(payload).encode('utf-8')

    # 4. Build the HTTP POST request
    request = urllib.request.Request(
        OLLAMA_URL, 
        data=data_bytes, 
        headers={'Content-Type': 'application/json'}
    )

    try:
        # 5. Send the request and wait for the response
        with urllib.request.urlopen(request) as response:
            # Read the raw byte response and decode it back into a Python dictionary
            response_data = json.loads(response.read().decode('utf-8'))
            
            # Extract the AI's actual text message
            ai_reply = response_data['message']['content']
            
            print(f"AI: {ai_reply}\n")
            
            # Add the AI's reply to the history so it remembers for next time
            chat_history.append({"role": "assistant", "content": ai_reply})
            
    except urllib.error.URLError as e:
        print(f"\n[Error] Could not connect to Ollama. Is it running? Details: {e}\n")