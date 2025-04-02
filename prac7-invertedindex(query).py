import os
import string
from collections import defaultdict

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    return text.split()

def build_inverted_index(document):
    inverted_index = defaultdict(set)
    for doc_id, text in document.items():
        words = preprocess_text(text)
        for word in words:
            inverted_index[word].add(doc_id)
    return inverted_index

def search(inverted_index, query):
    query_terms = preprocess_text(query)
    result_set = None
    for term in query_terms:
        if term in inverted_index:
            if result_set is None:
                result_set = inverted_index[term]
            else:
                result_set = result_set.intersection(inverted_index[term])
        else:
            return result_set()
    return result_set

documents = {
    1:"Information retrieval is an essential aspect of searchj engines.",
    2:"The field of information retrieval focuses ion algorithm.",
    3:"Search engines use retrieval techniques to improve perfromance.",
    4:"Deep learning models are used for information retrieval tasks.",
}

inverted_index = build_inverted_index(documents)

query = "retrieval"
result = search(inverted_index, query)

print(f"Documents containing the query '{query}': {sorted(result)}")

