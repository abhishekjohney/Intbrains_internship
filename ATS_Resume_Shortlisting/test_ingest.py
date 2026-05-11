import requests
import os
import json

BASE_URL = "http://localhost:8001"
RESUMES_DIR = "resumes"

def test_ingest():
    for filename in os.listdir(RESUMES_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(RESUMES_DIR, filename)
            print(f"Ingesting {filename}...")
            
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "application/pdf")}
                response = requests.post(f"{BASE_URL}/ingest", files=files)
                
            if response.status_code == 200:
                print(f"Success! {json.dumps(response.json(), indent=2)}")
            else:
                print(f"Failed! Status Code: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    test_ingest()
