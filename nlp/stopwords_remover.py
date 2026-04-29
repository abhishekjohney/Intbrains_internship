import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def remove_stopwords(text):
    """
    Tokenizes the text and removes common English stopwords.
    """
    words = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    
    # 3. Filter out the stopwords (converting to lowercase for accurate matching)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # 4. Rejoin the remaining words into a clean string
    cleaned_text = " ".join(filtered_words)
    
    return filtered_words, cleaned_text

if __name__ == "__main__":
    print("Please enter the text to remove stopwords from (press Enter when done):")
    user_input = input("> ")

    if user_input.strip():
        extracted_words, cleaned_string = remove_stopwords(user_input)

        print("\n--- Original Text ---")
        print(user_input)

        print("\n--- Cleaned Output (String) ---")
        
        print(cleaned_string)
        
        print("\n--- Cleaned Output (Word List) ---")
        print(extracted_words)
    else:
        print("\nNo text was entered. Please run the program again.")