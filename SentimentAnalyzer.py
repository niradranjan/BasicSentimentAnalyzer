#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:09:21 2017

@author: nirad
"""
import re
import nltk
from nltk.corpus import stopwords

cleaned_tweets_list = []
stop_word_removed_tweets_list = []
pos_score = 0
neg_score = 0
count_pos = 0
count_neg = 0
count_neut = 0
sentence_sentiment = 0
overall_score = 0
sentimet_dict = {}
counter = 0

tweet_file = open('/Users/nirad/Documents/Fintech_Twitter_Dump', 'r')
uncleaned_tweet_list = tweet_file.readlines()
tweet_file.close
#print uncleaned_tweet_list

positive_file = open('/Users/nirad/Documents/positive.txt', 'r')
positive_word_list =[]
contents_pos = positive_file.readlines()
for i in range(len(contents_pos)):
   positive_word_list.append(contents_pos[i].strip('\n'))
positive_file.close()
#print positive_word_list  

negative_file = open('/Users/nirad/Documents/negative.txt', 'r')
negative_word_list =[]
contents_neg = negative_file.readlines()
for i in range(len(contents_neg)):
   negative_word_list.append(contents_neg[i].strip('\n'))
negative_file.close() 
#print negative_word_list

increment_file = open('/Users/nirad/Documents/inc.txt', 'r')
incr_word_list =[]
contents_incr = increment_file.readlines()
for i in range(len(contents_incr)):
   incr_word_list.append(contents_incr[i].strip('\n'))
increment_file.close() 
#print incr_word_list

decrement_file = open('/Users/nirad/Documents/dec.txt', 'r')
decr_word_list =[]
contents_decr = decrement_file.readlines()
for i in range(len(contents_decr)):
   decr_word_list.append(contents_decr[i].strip('\n'))
decrement_file.close() 
#print decr_word_list

flip_file = open('/Users/nirad/Documents/flip.txt', 'r')
flip_word_list =[]
contents_flip = flip_file.readlines()
for i in range(len(contents_flip)):
   flip_word_list.append(contents_flip[i].strip('\n'))
flip_file.close() 
#print flip_word_list


for tweet in uncleaned_tweet_list:
   temp = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', tweet)
   temp = re.sub(r'&amp;|&amp;quot;|&amp;amp', '', temp)
   temp = re.sub(r'#', '', temp)
   temp = re.sub(r'RT ', '', temp)
   temp = re.sub(r'@[a-zA-Z0-9:]*', '', temp)
   temp = re.sub(r'\$[a-zA-Z0-9]*', '', temp)
   temp = re.sub(r'[0-9]*','',temp)
   temp = re.sub(r'\n','',temp)
   temp=temp.decode("utf8").encode('ascii','ignore')
   cleaned_tweets_list.append(temp)
#print cleaned_tweets_list

for tweet in cleaned_tweets_list:
   #print tweet
   tweet_stop_word_removed = ' '.join([word for word in tweet.split() if word not in (stopwords.words('english'))])
   #print tweet_stop_word_removed
   #print "############"
   stop_word_removed_tweets_list.append(tweet_stop_word_removed)
#print stop_word_removed_tweets_list

for tweet_sentence in stop_word_removed_tweets_list:
   #print tweet_sentence
   pos_score = 0
   neg_score = 0
   tokens = nltk.word_tokenize(tweet_sentence)
   #print tokens
   for t1 in tokens:
       for pw in positive_word_list:
           if t1.lower() == pw:
               indx1 = tokens.index(t1) - 1
               if tokens[indx1].lower() in incr_word_list:
                   pos_score = pos_score + 2
               elif tokens[indx1].lower() in decr_word_list:
                   pos_score = pos_score + 0.5
               elif tokens[indx1].lower() in flip_word_list:
                   neg_score = neg_score - 1    
               else:    
                   pos_score = pos_score + 1
   #print pos_score           
   for t2 in tokens:
       for nw in negative_word_list:
           if t2.lower() == nw:
               indx2 = tokens.index(t2) - 1
               if tokens[indx2].lower() in incr_word_list:
                   neg_score = neg_score - 2
               elif tokens[indx2].lower() in decr_word_list:
                   neg_score = neg_score - 0.5
               elif tokens[indx2].lower() in flip_word_list:
                   pos_score = pos_score + 1   
               else:    
                   neg_score = neg_score - 1
   #print neg_score           
   sentence_sentiment = neg_score + pos_score
   #print sentence_sentiment
   sentimet_dict[tweet_sentence] = sentence_sentiment
   overall_score = overall_score + sentence_sentiment

   if sentence_sentiment > 0:
       count_pos = count_pos + 1
   elif sentence_sentiment < 0:    
       count_neg = count_neg + 1
   else:
       count_neut = count_neut + 1
   counter = counter + 1
   print counter     

print sentimet_dict
print count_pos
print count_neg
print count_neut
print overall_score