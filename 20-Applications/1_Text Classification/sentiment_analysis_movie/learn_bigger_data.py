#define a tokenizer function that cleans the unprocessed text data from
#movie_data.csv file that we constructed in the beginning of this chapter
#and separates it into word tokens while removing stop words

import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text = re.sub('[\W]+', ' ', text.lower())+ ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

#define a generator function,stream_docs£¬that reads in and returns one document at a time
def stream_docs(path):
    with open(path,'r') as csv:
        next(csv) #skip header
        for line in csv:
            text,label = line[:-3],int(line[-2])
            yield text,label

#test stream_docs
next(stream_docs(path = 'F:/developSamples/ml/aclImdb/movie_data.csv'))

#define a function ,get_minibatch , that will take a document stream from the
#stream_docs function and return a particular number of documents specified by
#the size parameter
def get_minibatch(doc_stream,size):
    docs,y = [],[]
    try:
        for _ in range(size):
            text,label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y

#cann't use the CountVectorizer for out-of-core learning, the TfidfVectorizer needs to 
#keep the all feature vectors of the training dataset in memory to calculate the inverse
#document frequencies.Anohter useful vectorizer for text processing implemented in scikit-learn
#is HashingVectorizer.HashingVectorizer is data-independent and make use of teh Hashing trick via the 32-bit
#MurmurHash3 algorithm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error = 'ignore',n_features = 2**21,preprocessor = None,tokenizer= tokenizer)
clf = SGDClassifier(loss='log',random_state = 1,n_iter = 1)
doc_stream = stream_docs(path = 'F:/developSamples/ml/aclImdb/movie_data.csv')

#start the out-of-core learning using the following code:
import pyprind 
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])
for _ in range(45):
    X_train,y_train = get_minibatch(doc_stream,size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train,y_train,classes = classes)
    pbar.update()


#use the last 5000 documents to evaluate the performance of our model
X_test,y_test = get_minibatch(doc_stream,size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test,y_test))

