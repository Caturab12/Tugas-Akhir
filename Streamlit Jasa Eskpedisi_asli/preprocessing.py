import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')
nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(tweet):
  #menghapus username twiter
  text=re.sub('@[^\s]+',' ', tweet)
  #menghapus https dan http
  text=re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
  #menghilangkan tanda baca
  text = text.translate(str.maketrans(' ', ' ',string.punctuation))
  #mengganti karakter html dengan tanda petik
  text=re.sub('[^a-zA-Z]',' ',text)
  #mengganti line baru dengan spasi
  text = re.sub("\n"," ",text)
  #mengubah ke huruf kecil
  text = text.lower()
  #menghapus single char
  text = re.sub(r"\b[a-zA-Z]\b"," ",text)
  #memisahkan dan menggabungkan kata
  text=' '.join(text.split())
  return text

normalized_word = pd.read_csv("Normalisasi.csv", encoding='latin1')

normalized_word_dict={}
for index, row in normalized_word.iterrows():
  if row[0] not in normalized_word_dict:
    normalized_word_dict[row[0]] = row[1]

def normalized_term(document):
  return [normalized_word_dict[term] if term in normalized_word_dict else term for term in document]


stopword = stopwords.words('indonesian')
txt_stopword = pd.read_csv("stopwords.txt", names=["stopwords"], header = None)

stopword.extend(['wkwkwk', 'ke', 'amp','diginii'])

stopword.extend(txt_stopword["stopwords"][0].split(' '))
stopword = set(stopword)

def stopwords(tweet):
  tweet = [word for word in tweet if word not in stopword]
  return tweet

# stemming
def stem_text(text):
    text = " ".join(text)
    text = stemmer.stem(text)
    return text.split()

def join_text(text):
    text = ' '.join(text)
    return text