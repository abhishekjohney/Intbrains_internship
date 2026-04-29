import spacy
import re
import os

# 1. INITIALIZE THE AI
print("Booting up NLP Engine (spaCy)...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: NLP model not found. Please run: python -m spacy download en_core_web_sm")
    exit()

os.system('cls' if os.name == 'nt' else 'clear')

# 2. THE EXTRACTION & REDACTION LOGIC
def scan_for_sensitive_data(text):
    """Scans text, extracts PII, and returns tokens, alerts, and redacted text."""
    doc = nlp(text)
    
    # 1. Gather Tokens & POS Tags (We return them now instead of printing here)
    pos_debug_list = [f"{token.text} [{token.pos_}]" for token in doc]
    
    flagged_items = []
    
    # A. Whitelist for False Positives
    whitelist = [
        "AI", "NLP", "OSError", "GPE", "Regex", "spaCy", "API", 
        "THE EXTRACTION LOGIC", "AI Context and Regex Patterns"
    ]
    
    # B. Contextual AI Search (spaCy)
    target_labels = {
        "PERSON": "Name",
        "ORG": "Company/Organization",
        "GPE": "Location",
        "DATE": "Date/Time"
    }
    
    for ent in doc.ents:
        if ent.label_ in target_labels and ent.text not in whitelist:
            flagged_items.append({
                "value": ent.text, 
                "type": target_labels[ent.label_]
            })

    # C. Strict Pattern Search (Regex)
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    for email in emails:
        flagged_items.append({"value": email, "type": "Email Address"})
        
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    for phone in phones:
        flagged_items.append({"value": phone, "type": "Phone Number"})

    # D. The Redaction Engine
    redacted_text = text
    sorted_items = sorted(flagged_items, key=lambda x: len(x['value']), reverse=True)
    
    for item in sorted_items:
        replacement_tag = f"[REDACTED: {item['type'].upper()}]"
        redacted_text = redacted_text.replace(item['value'], replacement_tag)

    # Return all three pieces of data to the main function
    return pos_debug_list, flagged_items, redacted_text


# 3. THE TERMINAL INTERFACE (3-STEP AUDIT MODE)
def main():
    print("------------------------------------------------------------")
    print("ENTERPRISE PII DETECTION SCANNER ONLINE")
    print("------------------------------------------------------------")
    print("Paste your text below. Type 'SCAN' on a new line to process, or 'exit' to quit.\n")
    
    while True:
        print("Raw Input:")
        
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
                
            if line.strip().lower() in ['exit', 'quit']:
                print("Shutting down scanner...")
                return
                
            if line.strip().upper() == 'SCAN':
                break
                
            lines.append(line)
            
        user_input = "\n".join(lines)
        
        if not user_input.strip():
            continue
            
        # Run the scan and unpack the three returned variables
        pos_tags, found_data, safe_text = scan_for_sensitive_data(user_input)
        
        # --- PHASE 1: CLASSIFICATION AND TOKENS ---
        print("\n============================================================")
        print("1. CLASSIFICATION AND TOKENS")
        print("============================================================")
        print(f"{pos_tags}\n")
        
        if not found_data:
            print("Status: Clean (No PII detected)\n")
            print("------------------------------------------------------------\n")
            continue
            
        print("============================================================")
        print("2. PII DETECTED IN PAYLOAD")
        print("============================================================")
        for item in found_data:
            print(f" -> Found: '{item['value']}' (Identified as: {item['type']})")
        print()
            
        print("============================================================")
        print("3. SAFE TEXT (REDACTED DOCUMENT)")
        print("============================================================")
        print(safe_text)
        print("\n------------------------------------------------------------\n")

if __name__ == "__main__":
    main()