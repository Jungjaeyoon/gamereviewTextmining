# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk;import pandas as pd;import numpy as np
from nltk.stem.snowball import SnowballStemmer;from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer; from sklearn import linear_model
from sklearn.cross_validation import KFold
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from matplotlib import pyplot as plt; import lda
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn


path="C:/Users/arago/Python/GameText/"
frame="user_review_data_list.csv"
X=pd.read_csv(path+frame,encoding='cp949')
X= pd.read_csv(path+'Review_Data_Critic.csv',encoding='cp949')

sens=[nltk.tokenize.sent_tokenize(X['review'][i]) for i in range(0,len(X['review']))]

names=set(X['title'])
[ x.split(' ') for x in names]
names_words = []
for x in names:
    names_words += [ word.lower() for word in x.split() ]    

import collections
cnt = collections.Counter()
for x in names_words:
    cnt[x] += 1



tokens=[]
for i in range(0,len(sens)):
    token=[]
    for j in range(0,len(sens[i])):
        token+=nltk.tokenize.word_tokenize(sens[i][j])
    tokens.append(list(token))
pos=[]
for t in tokens:
    pos_tokens=[token for token, pos in nltk.pos_tag(t) if pos.startswith('RB')|pos.startswith('JJ')]
    pos.append(pos_tokens)

    
collections.OrderedDict(sorted(cnt.items(), key=lambda t: t[1]))


#removal stopwords
stop=nltk.corpus.stopwords.words('english')
stop+=["!","...",")","(","/",".",",","?","-","''","``","'d",":",";","***","*","%","$","@","#","&","+","~","'s","n't","'m","'d"]
additionalstop = ['game','make','un', 'es', 'juego', 'la', 'el', 'con', 'lo', 'los', 'para', 'una', 'si', 'se', 'por', 'le']
stop+= additionalstop
stop += names_words



pos2=[]
for t in tokens:
    pos_tokens=[token for token, pos in nltk.pos_tag(t) if pos.startswith('NN')|pos.startswith('VERB')]
    pos2.append(pos_tokens)



stemmer2=SnowballStemmer("english",ignore_stopwords=True)
sin_snowball=[]
for p in pos2:
    singels=[stemmer2.stem(p[i]) for i in range(0,len(p))]
    sin_snowball.append(singels)




sin_snowball2=[]
for singles in sin_snowball:
    singles2=[word for word in singles if word not in stop]
    singles2=[a for a in singles2 if len(a)!=1] #한 글자 지우기 
    sin_snowball2.append(singles2)


all_words2=[]
for doc in sin_snowball2:
    all_words2+=[word for word in doc]

all_words_raw = []
for doc in pos2:
    all_words_raw +=[x for x in doc if x not in stop]


#frequency analysis
fd=nltk.FreqDist(all_words2)
fd_table=pd.DataFrame(np.array(fd.most_common(len(set(all_words2)))))
fd_table[1]=fd_table[1].apply(pd.to_numeric)
fd_table=fd_table[fd_table[1]>=100]

#fd_table.to_csv('frequency_table_nopreprocessed_cr.csv')
#remove words
sin_snowball3=[]
for singles in sin_snowball2:
    singles2=[word for word in singles if word in list(fd_table[0])]
    sin_snowball3.append(singles2)

#clean doctument
doc2=[]
for singeles  in sin_snowball3:
    result =  " ".join(singeles)
    doc2.append(result)
countvec=CountVectorizer()
tf_lda=countvec.fit_transform(doc2)
topic_X=tf_lda.toarray()
vocab=countvec.get_feature_names() ####################sen-topic에서 topic 검색 단어로 쓰임

#########################전처리 끝(RB,JJ)#######################################

countvec=CountVectorizer()
tf_lda=countvec.fit_transform(doc2)
topic_X=tf_lda.toarray()
vocab=countvec.get_feature_names() ####################sen-topic에서 topic 검색 단어로 쓰임


for x in [3,4,5,6,7]:
    for s in [0.1]:
        model=lda.LDA(n_topics= x ,n_iter=500,random_state=6,alpha = s)
        model.fit(topic_X)
        topic_word=model.topic_word_
        n_top_words=20
        lda_results = []
        for i, topic_dist in enumerate(topic_word):
            topic_words=np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
            print('Topic',i,topic_words)
            lda_results.append([i,topic_words])
        lda_results = pd.DataFrame(lda_results,columns = ['Topic_N','Words'])
        lda_results.to_csv('nwords_50_CLDA_results_NNVERB%s_%s.csv' % (x, s))




