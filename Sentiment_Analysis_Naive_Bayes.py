#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:48:56 2017

@author: saheli06
"""

import pandas as pd
import pprint, pickle
import numpy as np
import sklearn.utils
import random

data = pd.read_csv("Final_Data.csv")

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import unicodedata
import string
tokenizer = RegexpTokenizer(r'\w+')
punctuation = list(string.punctuation)
corpus = []
for i in range(len(data)):
	if  i == 147079: break
    review = re.sub('[^a-zA-Z]', ' ', data['tweet'][i])
    review = review.lower()
#       unicodedata.normalize('NFKD', review).encode('ascii','ignore')
    review = review.encode('utf8')
    r = review
    review = tokenizer.tokenize(review)
    review = [str(x) for x in review]
    #ps = PorterStemmer()
    review = [word for word in review if not word in set(stopwords.words('english') + punctuation + ['rt', 'https','via', 'co'])]
    review = [str(x) for x in review]
    review = ' '.join(review)
    corpus.append(review)

positive_vocabs = open("positive.txt","r").read()
negative_vocab = open("negative.txt","r").read()
documents = []
positive_words = positive_vocabs.split('\n')
negative_words = negative_vocab.split('\n')

for r in positive_words:
    documents.append( (r, "pos") )
for r in negative_words:
    documents.append( (r, "neg") )
#Frequency Distribution
from nltk.tokenize import word_tokenize
all_words = []
words = []
short_pos_words = word_tokenize(positive_vocabs)
short_neg_words = word_tokenize(negative_vocab)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
	
featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class SentimentClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        senti = []
        for c in self._classifiers:
            v = c.classify(features)
            senti.append(v)
        return mode(senti)

    def confidence(self, features):
        senti = []
        for c in self._classifiers:
            v = c.classify(features)
            senti.append(v)

        choice_votes = senti.count(mode(senti))
        conf = choice_votes / len(senti)
        return conf

training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

Sentiment_Classifier = SentimentClassifier(classifier)
 
def sentiment(text):
    feats = find_features(text)
    return Sentiment_Classifier.classify(feats),Sentiment_Classifier.confidence(feats)

sentiemnt = []
for i in range(len(corpus)):
    tweet = corpus[i]
    s = sentiment(tweet)
    sentiemnt.append(s)
# Co- Occurance matrix for tweets
import operator 
from collections import Counter
from collections import defaultdict
com = defaultdict(lambda : defaultdict(int))
count_all = Counter()
for i in range(len(corpus)):
    for word in corpus[i]:
        tweet = corpus[i]
        tweet = tweet.split()
        # Create a list with all the terms
        terms_all = [term for term in tweet]
        # Build co-occurrence matrix
        for i in range(len(terms_all)-1):            
            for j in range(i+1, len(terms_all)):
                w1, w2 = sorted([terms_all[i], terms_all[j]])                
                if w1 != w2:
                    com[w1][w2] += 1
# Creating 20 most common used words                    
com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in com:
    t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print(terms_max[:30])

# Creating 20 most common used words                    
com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in com:
    t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print(terms_max[:30])

