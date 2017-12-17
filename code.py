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

nltk.download()


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

"""

#stemming
stemmer2=SnowballStemmer("english",ignore_stopwords=True)
singles_snowball=[]
for p in pos:
    singels=[stemmer2.stem(p[i]) for i in range(0,len(p))]
    singles_snowball.append(singels)


#ntoken = [stemmer2.stem(x) for x in ntokens]

#removal stopwords
stop=nltk.corpus.stopwords.words('english')
stop+=["!","...",")","(","/",".",",","?","-","''","``","'d",":",";","***","*","%","$","@","#","&","+","~","'s","n't","'m","'d"]
additionalstop = ['game','make','un', 'es', 'juego', 'la', 'el', 'con', 'lo', 'los', 'para', 'una', 'si', 'se', 'por', 'le']
stop+= additionalstop
title = []
for name in names:
    title += (nltk.tokenize.word_tokenize(name))
title = [stemmer2.stem(x) for x in title]

stop+= title


singels_snowball2=[]
for singles in singles_snowball:
    singles2=[word for word in singles if word not in stop]
    singles2=[a for a in singles2 if len(a)!=1] #한 글자 지우기 
    singels_snowball2.append(singles2)


#30번 이하 단어 지우기
all_words=[]
for doc in singels_snowball2:
    all_words+=[word for word in doc]

fd=nltk.FreqDist(all_words)
fd_table=pd.DataFrame(np.array(fd.most_common(len(set(all_words)))))
fd_table[1]=fd_table[1].apply(pd.to_numeric)
fd_table=fd_table[fd_table[1]>=30]
singels_snowball3=[]
print('1')
for singles in singels_snowball2:
    singles2=[word for word in singles if word in list(fd_table[0])]
    singels_snowball3.append(singles2)

print('2')
#최종 클렌징된 문서 셋
doc=[]
for singeles  in singels_snowball3:
    result =  " ".join(singeles)
    doc.append(result)

print('3')
"""


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




#sentiment score using lasso (형, 부 - 1981개 단어)

#===searching parameter===#
#len(countvectorizer.get_feature_names())
#kf = KFold(words.shape[0], n_folds = 10)
# within 다른 range
#alphas = list(np.arange(0.0001,0.0022,0.0002))+[0.005,0.01,0.02,0.05,1,5,10]
#e_alphas = list()
#e_alphas_r = list()  #holds average r2 error
#for alpha in alphas:
#    lasso = linear_model.Lasso(alpha=alpha)
#    err = list()
#    err_2 = list()
#    for tr_idx, tt_idx in kf:
#        X_tr , X_tt = words.loc[tr_idx], words.loc[tt_idx]
#        y_tr, y_tt = score[tr_idx], score[tt_idx]
#        lasso.fit(X_tr, y_tr)
#        y_hat = lasso.predict(X_tt)
#        err_2.append(lasso.score(X_tt,y_tt)) #R^2
#        err.append(np.average((y_hat - y_tt)**2)) #SSE
#        
#    e_alphas.append(np.average(err))
#    e_alphas_r.append(np.average(err_2))
#    print(alpha)
#plt.plot(alphas[:-5], e_alphas_r[:-5])
#plt.title("Best lamda",fontsize=15)
#plt.xlabel('lamda',fontsize=15)
#plt.ylabel('R^2',fontsize=15)
#plt.plot(alphas[:-5], e_alphas[:-5])
#plt.title("Best lamda",fontsize=15)
#plt.xlabel('lamda',fontsize=15)
#plt.ylabel('MSE',fontsize=15)

#sentiment score by words





#==============================================================================#
###########################감정 사전 구축 끝#####################################
#==============================================================================#
=====================================================================#
######################lDA끝#####################################################
#==============================================================================#

#==============================================================================#
#################sen-topic 문서 셋 구축(pos- NN,RB, VB, JJ)######################
#==============================================================================#
#POS
pos=[]
for t in tokens:
    pos_tokens=[token for token, pos in nltk.pos_tag(t) if pos.startswith('RB')|pos.startswith('JJ')|pos.startswith('NN')|pos.startswith('VB')]
    pos.append(pos_tokens)

#stemming
stemmer2_senti=SnowballStemmer("english",ignore_stopwords=True)
singles_snowball_senti=[]
for p in pos:
    singels_senti=[stemmer2_senti.stem(p[i]) for i in range(0,len(p))]
    singles_snowball_senti.append(singels_senti)
    
    
#removal stopwords



np.savetxt(path+"game_topic_sen.csv",game_topic,delimiter = ",")


singels_snowball2_senti=[]
for singles in singles_snowball_senti:
    singles2_senti=[word for word in singles if word not in stop]
    singles2_senti=[a for a in singles2_senti if len(a)!=1] #한 글자 지우기 
    singels_snowball2_senti.append(singles2_senti)

all_words_senti=[]
for doc in singels_snowball2_senti:
    all_words_senti+=[word for word in doc]



#frequency analysis
fd_senti=nltk.FreqDist(all_words_senti)
fd_table_senti=pd.DataFrame(np.array(fd_senti.most_common(len(set(all_words_senti)))))
fd_table_senti[1]=fd_table_senti[1].apply(pd.to_numeric)
fd_table_senti=fd_table_senti[fd_table_senti[1]>=10]
plot = fd.plot(100,cumulative=False)


#remove words
snowball3_senti=[]
for singles in singels_snowball2_senti:
    singles2_senti=[word for word in singles if word in list(fd_table_senti[0])]
    snowball3_senti.append(singles2_senti)

#clean doctument
all_doc=[]
for singeles  in snowball3_senti:
    result =  " ".join(singeles)
    all_doc.append(result)
 
countvectorizer=CountVectorizer()
tf=countvectorizer.fit_transform(all_doc)

words=pd.DataFrame(tf.toarray())
score=X['score']


import math
for x in range(len(score)):
    if math.isnan(score[x]) == True:
        score[x] = 0
   
final_lasso = linear_model.Lasso(alpha=0.0005)
final_lasso.fit(words, score)

     
      
fea_score=[[feature,coef] for feature, coef in zip(list(countvectorizer.get_feature_names()),list(final_lasso.coef_))]
fea_score=pd.DataFrame(np.array(fea_score))
fea_score.columns=['feature','sen_score']
fea_score['sen_score']=pd.to_numeric(fea_score['sen_score'])
fea_score=fea_score[(fea_score['sen_score']>0)|(fea_score['sen_score']<0)]

sentiment_list=list(fea_score['feature'])

   
# Process - 감정단어를 검색하고 앞뒤 n개 단어를 searching 
n=3
dws=[]
for d, docu in enumerate(snowball3_senti):
    for plo in sentiment_list:
        plo_score=list(fea_score[fea_score['feature']==plo]['sen_score'])[0]
        plo_idx=[i for i, w in enumerate(docu) if w==plo]
        for idx in plo_idx:
            s_idx=np.where(idx-n<0,0,idx-n)
            e_idx=np.where(idx+n+1>len(docu),len(docu),idx+n+1)
            f_ngram=docu[s_idx:idx]
            b_ngram=docu[idx+1:e_idx]

            if len(f_ngram)!=0:
                topic_idx=[i for i, w in enumerate(f_ngram) if w in vocab]
                if len(topic_idx)!=0:
                    topic_words=f_ngram[np.max(topic_idx)]
                    twi=vocab.index(topic_words)
                    dws.append([d,twi,plo_score])
            elif len(b_ngram)!=0:
                topic_idx=[i for i, w in enumerate(b_ngram) if w in vocab]
                if len(topic_idx)!=0:
                    topic_words=b_ngram[np.min(topic_idx)]
                    twi=vocab.index(topic_words)
                    dws.append([d,twi,plo_score])
            else:
                next
    print(d)
           
dwsm= np.zeros(shape=(d+1,len(vocab))) 
for i in range(0,len(dws)):
    dwsm[dws[i][0]][dws[i][1]]=dwsm[dws[i][0]][dws[i][1]]+dws[i][2]

np.savetxt(path+"dwsm.csv",dwsm,delimiter = ",")

#==============================================================================#
##################### sen-topic 구축 끝(dwsm) ############################## 
#==============================================================================#

#==============================================================================#
############################ topic별 감정점수 계산############################### 
#==============================================================================#
#전체 토픽별 점수
np.dot(dwsm,topic_word.T).sum(axis=0)

#게임/토픽별 점수
grouped=X.groupby(['title'])
game_num=[0]+list(grouped.last().sort('id')['id']+1)
game_num = list(X.index)
game_topic=[]
for i in range(0,len(game_num)-1):
    if game_num[i+1] <= d+1:
        s_num=game_num[i]
        f_num=game_num[i+1]
        game_topic.append(list(np.dot(dwsm[s_num:f_num],topic_word.T).sum(axis=0)))
    else:
        next

np.savetxt(path+"game_topic_sen.csv",game_topic,delimiter = ",")




topic = pd.read_csv(path+'game_topic_sen.csv', names = ['Design','Gameplay','Story','Multiplay','Remaster','Series','7'])

X['title']
