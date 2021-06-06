from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
text=input()
ps = PorterStemmer()
review=text.lower()
nltk.download('stopwords')
tokens = word_tokenize(review)
words = [word for word in tokens if word.isalpha()]
review = [ps.stem(word) for word in words if not word in stopwords.words('english')]
review = ' '.join(review)
voc_size=10000
onehot_repr=[one_hot(review,voc_size)] 
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print([embedded_docs])