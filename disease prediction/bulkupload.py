import os
import requests

# The URL of your FastAPI route
API_URL = "http://localhost:8000/api/learn-disease"

# The folder where all your text files live
FOLDER_PATH = "./diseases"

def upload_all_files():
    # Make sure the folder exists
    if not os.path.exists(FOLDER_PATH):
        print(f"Error: Could not find the folder '{FOLDER_PATH}'")
        return

    print("Starting bulk ingestion...\n")
    success_count = 0

    # Loop through every file in the folder
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".txt"):
            # Get the full, absolute path of the file (which FastAPI needs)
            absolute_path = os.path.abspath(os.path.join(FOLDER_PATH, filename))
            
            # Prepare the JSON payload exactly like we did in Swagger
            payload = {"file_path": absolute_path}
            
            try:
                # Fire the file to the API
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    print(f"✅ Successfully memorized: {filename}")
                    success_count += 1
                else:
                    print(f"❌ Failed to ingest {filename}: {response.text}")
            except Exception as e:
                print(f"⚠️ Error connecting to server for {filename}: {str(e)}")

    print(f"\nDone! Successfully ingested {success_count} diseases into Qdrant.")

if __name__ == "__main__":
    upload_all_files()