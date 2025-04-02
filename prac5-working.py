from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


with open('text1.txt', "r") as f:
    words = f.read()

words_list = words.split()
filtered_sentence = []


with open('to_write_file.txt', 'w') as write_file:
    for w in words_list:
        if w.lower() not in stop_words:
            filtered_sentence.append(w)
            write_file.write(w + " ")  

print("Filtered words written to 'to_write_file.txt'")
