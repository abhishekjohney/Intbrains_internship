import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def tokenize_paragraph(paragraph):
    """
    Tokenizes a given paragraph into sentences and words.
    """
    sentences = sent_tokenize(paragraph)
    
    words = word_tokenize(paragraph)

    return sentences, words

if __name__ == "__main__":
    print("Please enter the paragraph you want to tokenize (press Enter when done):")
    user_input = input("> ")

    # Check if the user actually entered something
    if user_input.strip():
        # Run the function
        extracted_sentences, extracted_words = tokenize_paragraph(user_input)

        # Display the results
        print("\n--- Sentences ---")
        for i, sentence in enumerate(extracted_sentences, 1):
            print(f"{i}. {sentence}")

        print("\n--- Words ---")
        print(extracted_words)
        print(f"\nTotal Word Count: {len(extracted_words)}")
    else:
        print("\nNo text was entered. Please run the program again.")