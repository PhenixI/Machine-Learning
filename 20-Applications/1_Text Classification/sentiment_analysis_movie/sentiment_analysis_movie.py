#data prepare, read the movie reviews into a pandas DataFrame Object,dataset sources: http://ai.stanford.edu/~amaas/data/sentiment/
#import pyprind
import pandas as pd
#import os
#pbar = pyprind.ProgBar(50000)
#labels = {'pos':1,'neg':0}
#df = pd.DataFrame()
#for s in ('test','train'):
#    for l in ('pos','neg'):
#        path = 'F:/developSamples/ml/aclImdb/%s/%s' % (s,l)
#        for file in os.listdir(path):
#            with open(os.path.join(path,file),'r',encoding = 'utf-8') as infile:
#                txt = infile.read()
#            df = df.append([[txt,labels[l]]],ignore_index = True)
#            pbar.update()

#df.columns = ['review','sentiment']

#shuffle DataFrame using the permutation function from the np.random submodule
#store the assembled and shuffled movie review dataset as a CSV file
#import numpy as np
#np.random.seed(0)

#df = df.reindex(np.random.permutation(df.index))
#df.to_csv('F:/developSamples/ml/aclImdb/movie_data.csv',index=False)

df = pd.read_csv('F:/developSamples/ml/aclImdb/movie_data.csv',encoding = 'gbk')
#df.head(3)

#import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
#count = CountVectorizer()
#docs = np.array(['The sun is shining','The weather is sweet','The sun is shining and the weather is sweet'])
#bag = count.fit_transform(docs)

##the vocabulary is stored in a python dictionary, which map the unique words that are mapped to integer indices
#print (count.vocabulary_)
##print the feature vectors 
#print(bag.toarray())

#from sklearn.feature_extraction.text import TfidfTransformer
#tfidf = TfidfTransformer()
#np.set_printoptions(precision = 2)
#print (tfidf.fit_transform(bag).toarray())

#clean text data.regex (regular expression)

import re
def preprocessor(text):
    #try to remove the entire HTML markup that was contained in the moive reviews
    text = re.sub('<[^>]*>', '', text)
    #find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    #remove all non-word characters form the text via the regex [\W]+
    #convert the text into lowercase characters, add the emoticions to the end of the processed document string
    text = re.sub('[\W]+', ' ',text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

#preprocessor(df.loc[2, 'review'])
#preprocessor("</a>This :) is :( a test :-)!")
df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

#tokenizer_porter('runners like running and thus they run')

#import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

#divide the DataFrame of cleaned text documents into 25000 documents for training and 25000 documents for testing
X_train = df.loc[:25000,'review'].values
y_train = df.loc[:25000,'sentiment'].values
X_test = df.loc[25000:,'review'].values
y_test = df.loc[25000:,'sentiment'].values

#use a GridSearchCV object to find the optimal set parameters for your logistic regression
#model using 5-fold stratified cross-validation
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
#combine the CountVectorizer and TfidfTransformer
tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
#in the first dictionary, we used the TfidfVectorizer with its default settings (use_idf = True,smooth_idf = True, and norm = 'l2')
#to calculate the tf-idfs, in the second dictionary, we set those parameters to use_idf = False,smooth_idf = False, and norm = None
#in order to train a model based on raw term frequencies. Furthermore, for the logistic regression classifier
#itself, we trained models using L2 and L1 regularization via the penalty parameter and compared different regularization strengths by
#defining a range of values for the inverse-regularization parameter C.
param_grid = [{'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer,tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer,tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}]
lr_tfidf = Pipeline([('vect', tfidf),('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,scoring='accuracy',cv=5, verbose=1,n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)

#print the best parameter set
print ('Best paramter set :%s' % gs_lr_tfidf.best_params_)


#print the 5-fold cross-validation accuracy scores on the training set 
#and the classification accuracy on the test dataset
print ('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print ('Test Accuracy: %.3f' % clf.score(X_test,y_test))
