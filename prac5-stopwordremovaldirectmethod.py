##Direct text
##Code:
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

example_sent = "This is a sample sentence, show off the stop words filtration."
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)
print(word_tokens)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
print(filtered_sentence)
filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
        print(filtered_sentence)


##reading text from a text file & importing it in a text file
##Code:
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
f = open('to_filt_text.txt', "r")
words = f.read()
words_list = words.split()
filtered_sentence = []
write_file = open('to_write_file.txt','w')
for w in words_list:
    if w not in stop_words:
        filtered_sentence.append(w)
        print(filtered_sentence)
        write_file.write(w+" ")

