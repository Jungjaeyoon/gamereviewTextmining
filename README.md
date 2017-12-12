# 게임 리뷰 토픽 모델링과 감정 분석
## Purpose
- ### 게임 제품의 사용자 리뷰가 어떤 주제로 이루어졌는가?
- ### 각 주제는 어떤 감정(긍/부정)을 지니고 있는가?
- ### 여러 게임들 각각의 주제와 감정 정도를 측정하여 보여줄 수 있도록 하기

* ## 순서
  1. #### 데이터 수집
  2. #### 토픽 모델링
  3. #### 감정 분석
  4. #### 분석 결과



* ## 폴더 설명
  * #### data_scrap : how to collect review data from metacritic.com
  * #### topic_modeling : topic modeling with LDA and word2vec
  * #### sentiment_analysis : sentiment scoring for each topic



## 1. Data Scrap
  ### scrap data from metacritic.com


## 2. Topic modeling
  ### 2.1 topic modeling using LDA
  #### Use NLTK, Sklearn
  #### 각 게임 제품 리뷰가 어떤 주제들로 구성되어있을까?
  - #### 토픽 생성을 위한 과정
    1. #### 기본 전처리 과정
      - #### 토큰화
        <pre><code> nltk.tokenize.sent_tokenize() </pre></code>
      - #### stemming
        nltk의 SnowballStemmer 사용
      - #### stop words 제거
        기본 stop words list에 추가로 제거할 단어 입력
         <pre><code>stop=nltk.corpus.stopwords.words('english') # basic stop words list
         stop+=["!","...",")","(","/",".",",","?","-","''","``","'d",":",";","***","*","%","$","@","#","&","+","~","'s","n't","'m","'d"] # puntuation, symbols
         additionalstop = ['game','make','un', 'es', 'juego', 'la', 'el', 'con', 'lo', 'los', 'para', 'una', 'si', 'se', 'por', 'le'] #해석불가한 단어들
         gametitle # 게임 제목
         stop+= additionalstop
         stop+= gametitle       
         </pre></code>

       - #### 길이, 빈도 기반 단어 제거
        - 길이가 1인 단어 제거
        - 단어 분포 확인 후 빈도가 낮은 단어 제거(100으로 설정)
        - 품사 기반 단어 선택
          - pos tagging을 통해 명사와 동사로 토픽 구성

          ![Alttext](sentiment_analysis/img/freqdist.png "word distribution")

    2. ### LDA 코드 실행
     #### 몇 개의 토픽으로 설정할 것인지 확인하기 위해 여러 토픽 수, alpha를 주며 결과 확인
     #### term-frequency matrix를 입력으로, 결과 해석을 위해 feature name을 추출해둠
    <pre><code>countvec=CountVectorizer()
    tf_lda=countvec.fit_transform(doc2)
    topic_X=tf_lda.toarray()
    vocab=countvec.get_feature_names() #sen-topic에서 topic 검색 단어로 쓰임
    </pre></code>

    #### topic 수를 3~7까지 조정하며 결과 확인

      <pre><code>
      for x in [3,4,5,6,7]:
          for s in [0.05,0.1,0.15,0.2]:
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
            lda_results.to_csv('LDA_results_JJNNPOS%s_%s.csv' % (x, s))
      </pre></code>       
    3. ### ㅇ
