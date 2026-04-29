import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import os

print("Booting up NLP Engines...")

try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    exit()

os.system('cls' if os.name == 'nt' else 'clear')

def clean_with_nltk(text):
    """Cleans text using NLTK and applies Porter Stemming."""
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    clean_tokens = [
        word for word in tokens 
        if word not in string.punctuation and word not in stop_words
    ]
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in clean_tokens]
    
    return stemmed_tokens

def clean_with_spacy(text):
    """Cleans text using spaCy and applies Lemmatization."""
    # 1. Process the text (spaCy handles tokenization automatically)
    doc = nlp(text)
    
    clean_tokens = []
    
    for token in doc:
        
        if not token.is_punct and not token.is_stop and token.text.strip():
            # 5. Lemmatization
            clean_tokens.append(token.lemma_.lower())
            
    return clean_tokens

def main():
    print("------------------------------------------------------------")
    print("LIVE TEXT CLEANING & PREPROCESSING ENGINE")
    print("------------------------------------------------------------")
    print("Type your text below. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Raw Input: ")
        
        # Check if the user wants to quit
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down engine...")
            break
            
        # Ignore empty submissions
        if not user_input.strip():
            continue
            
        nltk_result = clean_with_nltk(user_input)
        print("\n============================================================")
        print("ENGINE 1: NLTK (Stemming)")
        print("============================================================")
        print(f"{nltk_result}\n")
        
        spacy_result = clean_with_spacy(user_input)
        print("============================================================")
        print("ENGINE 2: spaCy (Lemmatization)")
        print("============================================================")
        print(f"{spacy_result}\n")
        print("------------------------------------------------------------\n")

if __name__ == "__main__":
    main()