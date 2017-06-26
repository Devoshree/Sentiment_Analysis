from __future__ import division
import math
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords # import the stopword list
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

pos_reviews= pd.read_csv('F:/Downloads/NewPositiveModified.txt',header=None, names=["Text"])
print pos_reviews
num_reviews = pos_reviews["Text"].size

neg_reviews= pd.read_csv('F:/Downloads/NewNegativeModified.txt',header=None, names=["Text"])
print neg_reviews

# Function for getting bag of words from text

def review_to_words(raw_review):
    negation=["not","doesn't","didn't","neither","don't"]
    tex=BeautifulSoup(raw_review,"lxml")
    letters_only=re.sub("[^a-zA-Z]",
                        " ",
                        tex.get_text())
    lower_case=letters_only.lower()
    words=lower_case.split()
    meaningful_words=[w for w in words if not w in negation ]
    return (" ".join (meaningful_words))

def review_text(raw_review):
    tex=BeautifulSoup(raw_review,"lxml")
    letters_only=re.sub("[^a-zA-Z]",
                        " ",
                        tex.get_text())
    lower_case=letters_only.lower()
    words=lower_case.split()
    meaningful_words=[w for w in words]
    return (" ".join (meaningful_words))

# list of positive words
# Initialize an empty list to hold the clean positive reviews
pos_list = []
neg_list = []

# Loop over each review; create an index i that goes from 0 to the length
# of the review list

for i in xrange( 0, num_reviews ):
    if pos_reviews["Text"][i]==1:
        p=p+1
    pos_list.extend(review_to_words(pos_reviews["Text"][i]).split())

pos = len(pos_list)
print pos

num_reviews = neg_reviews["Text"].size
# list of negative words
for i in xrange( 0, num_reviews ):
    neg_list.extend(review_to_words(neg_reviews["Text"][i]).split())

neg = len(neg_list)
print neg

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

pos_data_features = vectorizer.fit_transform(pos_list)
neg_data_features = vectorizer.fit_transform(neg_list)
import numpy as np

vocab = vectorizer.get_feature_names()
# Numpy arrays are easy to work with, so convert the result to an 
# array
pos_data_features = pos_data_features.toarray()
dist_pos = np.sum(pos_data_features, axis=0)

neg_data_features = neg_data_features.toarray()
dist_neg = np.sum(neg_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set

test = pd.read_csv('F:/Downloads/Test.txt', sep= "\t" , header=None, names=["Text", "sentiment"])
test["sentiment"][16]='0'
print test
length=len(test["Text"])
print length
#k= test['sentiment'].count()
#print k
result=[]
tpp=0
tnp=0

negation=["not","doesn't","didn't","neither","don't"]
for i in xrange(0, length):
    res_neg=0
    res_pos=0
    if(i+1)%1000 ==0:
        print (i+1)
    test_list=[]
    #res_list=[]
    test_list.extend(review_text(test["Text"][i]).split())
    #res_list=[w for w in test_list if not w in res_list]
    #lemma = WordNetLemmatizer()
    #lem = map(lemma.lemmatize, test_list)
    flag=0
    for word in test_list:
        if word in negation:
            flag=1
        if flag==1 and word in pos_list:
            res_pos = res_pos+ math.log((pos_list.count(word)+1)/pos)
            flag=0
        elif flag==1 and word in neg_list:   
            res_neg = res_neg+ math.log((neg_list.count(word)+1)/neg)
            flag=0
        elif flag==0:
            res_pos = res_pos+ math.log((pos_list.count(word)+1)/pos)
            res_neg = res_neg+ math.log((neg_list.count(word)+1)/neg)
    tpp = (math.log(pos/(pos+neg)) +res_pos)
    tnp = (math.log(neg/(pos+neg)) +res_neg)
    if tpp>tnp:
        result.append('1')
    elif tnp>tpp:
        result.append('0')
print len(result)
final = pd.DataFrame( data={"Text":test["Text"],"sentiment":result})
final.to_csv('Model.csv')
print accuracy_score(result, test["sentiment"])
import sys
f = open("Model.txt", 'w')
sys.stdout = f
print final
print accuracy_score(result, test["sentiment"])
f.close()



