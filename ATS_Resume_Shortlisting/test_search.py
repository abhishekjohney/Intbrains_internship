import requests
import json

BASE_URL = "http://localhost:8001"

def test_search(query: str):
    print(f"Testing search for: '{query}'")
    response = requests.post(f"{BASE_URL}/search", json={"query": query})
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} matches.")
        for idx, match in enumerate(results):
            print(f"\n--- Match {idx + 1} ---")
            print(json.dumps(match, indent=2))
    else:
        print(f"Failed to search: {response.status_code} {response.text}")

if __name__ == "__main__":
    queries = [
        "Java Developer with Spring Boot",
        "Python developer with exactly 3 projects",
        "Candidate with more than 5 years of experience",
        "Frontend developer who used React in a project"
    ]
    for q in queries:
        test_search(q)
        print("="*40)
