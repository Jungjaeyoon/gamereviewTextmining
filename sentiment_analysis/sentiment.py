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

#lasso reg for sentiment dictionary

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
