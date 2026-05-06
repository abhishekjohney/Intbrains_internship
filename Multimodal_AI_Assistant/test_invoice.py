#!/usr/bin/env python
"""Test script to upload invoice and see extraction results."""
import requests
import json

API_URL = "http://localhost:8000/api/learn"
IMAGE_PATH = r"D:\Intbrains\Intbrains_internship\Multimodal_AI_Assistant\invoice_template.png"

# Adjust path if needed - looking for tax invoice image
import os
if not os.path.exists(IMAGE_PATH):
    # Try alternative paths
    for alt_path in [
        r"D:\Intbrains\Intbrains_internship\Multimodal_AI_Assistant\bill.png",
        r"D:\Intbrains\Intbrains_internship\Multimodal_AI_Assistant\bill2.png",
        r"D:\Intbrains\Intbrains_internship\Multimodal_AI_Assistant\invoice.png",
    ]:
        if os.path.exists(alt_path):
            IMAGE_PATH = alt_path
            break

print(f"Testing with: {IMAGE_PATH}")
print(f"File exists: {os.path.exists(IMAGE_PATH)}\n")

payload = {
    "file_path": IMAGE_PATH
}

try:
    print("Sending request to API (this may take 2-3 minutes for LLaVA processing)...")
    response = requests.post(API_URL, json=payload, timeout=300)  # 5 minute timeout
    print(f"Status Code: {response.status_code}\n")
    
    result = response.json()
    print("="*80)
    print("EXTRACTED DATA:")
    print("="*80)
    print(json.dumps(result, indent=2))
    
    if "image_description" in result:
        print("\n" + "="*80)
        print("IMAGE DESCRIPTION (Full Text):")
        print("="*80)
        print(result["image_description"])
        
except Exception as e:
    print(f"Error: {e}")
