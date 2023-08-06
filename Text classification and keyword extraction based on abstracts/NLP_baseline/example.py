import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

train = pd.read_csv('train.csv')
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')

test = pd.read_csv('test.csv')
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')

train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
