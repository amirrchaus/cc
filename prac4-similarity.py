import string
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords  # Importing stopwords from NLTK

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Simple stopwords list from NLTK
STOPWORDS = set(stopwords.words('english'))  # Load the stopwords for the English language

# Tokenize text and preprocess it (remove punctuation, lowercase, etc.)
def process(file):
    with open(file, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Remove punctuation and split by spaces to get tokens
    translator = str.maketrans('', '', string.punctuation)
    tokens = raw.translate(translator).lower().split()

    # Remove stopwords
    filtered_tokens = [w for w in tokens if w not in STOPWORDS]

    # Count word frequencies
    count = Counter(filtered_tokens)
    return count

def cos_sim(a, b):
    # Calculate cosine similarity between two vectors
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def getSimilarity(dict1, dict2):
    # Get the list of all unique words from both dictionaries
    all_words_list = list(set(dict1.keys()).union(set(dict2.keys())))
    all_words_list_size = len(all_words_list)

    # Initialize vectors for both dictionaries
    v1 = np.zeros(all_words_list_size, dtype=int)
    v2 = np.zeros(all_words_list_size, dtype=int)

    # Fill the vectors with word frequencies
    for i, key in enumerate(all_words_list):
        v1[i] = dict1.get(key, 0)
        v2[i] = dict2.get(key, 0)

    # Return cosine similarity
    return cos_sim(v1, v2)

if __name__ == '__main__':
    # Use raw strings to avoid invalid escape sequence warnings
    dict1 = process(r"C:\Users\Amirrchaus\Desktop\IR\text1.txt")
    dict2 = process(r"C:\Users\Amirrchaus\Desktop\IR\text2.txt")
    dict3 = process(r"C:\Users\Amirrchaus\Desktop\IR\text3.txt")

    # Print the similarity between the three pairs of documents
    print("Similarity between text1 and text2:", getSimilarity(dict1, dict2))
    print("Similarity between text1 and text3:", getSimilarity(dict1, dict3))
    print("Similarity between text2 and text3:", getSimilarity(dict2, dict3))
