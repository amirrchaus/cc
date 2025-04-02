# Term-document incidence matrix (from the table in the image)
matrix = {
    "Antony and Cleopatra": [1, 1, 0, 0, 0, 0, 1],
    "Julius Caesar": [1, 1, 0, 1, 1, 1, 0],
    "The Tempest": [0, 0, 1, 1, 1, 1, 1],
    "Hamlet": [0, 1, 1, 1, 1, 0, 0],
    "Othello": [0, 1, 1, 1, 1, 1, 0],
    "Macbeth": [1, 0, 1, 0, 0, 0, 1],
}

# Predefined term vectors for each term
terms = {
    "Brutus": [1, 1, 0, 1, 1, 1, 0],
    "Caesar": [1, 1, 0, 1, 1, 1, 1],
    "Calpurnia": [0, 1, 0, 0, 0, 0, 0],
    "Antony": [1, 1, 0, 0, 0, 0, 1],
    "Cleopatra": [1, 0, 0, 0, 0, 0, 0],
    "mercy": [1, 0, 1, 1, 1, 1, 0],
    "worser": [1, 0, 1, 1, 1, 0, 0],
}

# Function to print the matrix and terms
def print_matrix_and_terms():
    print("\nDocument-Term Matrix (Boolean values):")
    print("Documents:", list(matrix.keys()))
    print("Matrix:")
    for i, document in enumerate(matrix.keys()):
        print(f"{document}: {matrix[document]}")
   
    print("\nTerms (Boolean values):")
    for term, vector in terms.items():
        print(f"{term}: {vector}")
    print("\n")

# Function to perform and explain a query step
def explain_operation(op, vector1, vector2=None):
    if op == "NOT":
        print(f"NOT {vector1}: {[~v & 1 for v in vector1]}")  # Flip bits (keep it in binary 0/1)
        return [~v & 1 for v in vector1]
    elif op == "AND":
        print(f"{vector1} AND {vector2}: {[a & b for a, b in zip(vector1, vector2)]}")
        return [a & b for a, b in zip(vector1, vector2)]
    elif op == "OR":
        print(f"{vector1} OR {vector2}: {[a | b for a, b in zip(vector1, vector2)]}")
        return [a | b for a, b in zip(vector1, vector2)]

# Function to process a query
def process_query(query):
    query = query.strip()
    query_parts = query.split()
    result = None

    print("\nQuery Processing Steps:")
    i = 0
    while i < len(query_parts):
        term = query_parts[i]
        if term in terms:
            current_vector = terms[term]
            print(f"Term '{term}': {current_vector}")
            if result is None:
                result = current_vector
            else:
                operator = query_parts[i - 1]
                if operator == "AND":
                    result = explain_operation("AND", result, current_vector)
                elif operator == "OR":
                    result = explain_operation("OR", result, current_vector)
        elif term == "NOT" and i + 1 < len(query_parts):
            next_term = query_parts[i + 1]
            if next_term in terms:
                not_vector = explain_operation("NOT", terms[next_term])
                if result is None:
                    result = not_vector
                else:
                    result = explain_operation("AND", result, not_vector)
            i += 1  # Skip the term after NOT
        i += 1

    print("\nFinal result vector:", result)
    return result

# Function to display matching documents
def get_matching_documents(result_vector):
    documents = list(matrix.keys())
    matching_docs = [doc for doc, match in zip(documents, result_vector) if match]
    return matching_docs

# Main loop for user input
print("Boolean Retrieval System with Explanations")
print_matrix_and_terms()  # Print matrix and terms at the start
print("Enter queries using terms and operators (AND, OR, NOT). Example: Brutus AND Caesar AND NOT Calpurnia")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("Enter your query: ")
    if user_query.lower() == "exit":
        print("Exiting the system. Goodbye!")
        break
    try:
        result_vector = process_query(user_query)
        matching_documents = get_matching_documents(result_vector)
        print("\nMatching documents:", matching_documents if matching_documents else "No matches found.")
    except Exception as e:
        print("Error processing query:", str(e))
