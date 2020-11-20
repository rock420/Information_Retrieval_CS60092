import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import string
import os
import sys
import math
from heapq import nsmallest
import numpy as np

n = len(sys.argv)
if n<2:
    data_directory = "./dataset/"
    output_file = "result.txt"
else:
    data_directory = sys.argv[1]
    output_file = sys.argv[2]


class1_train = [os.path.join(data_directory,'class1','train',f) for f in os.listdir(os.path.join(data_directory,'class1','train')) ]
class2_train = [os.path.join(data_directory,'class2','train',f) for f in os.listdir(os.path.join(data_directory,'class2','train')) ]
class1_test =  [os.path.join(data_directory,'class1','test',f) for f in os.listdir(os.path.join(data_directory,'class1','test')) ]
class2_test =  [os.path.join(data_directory,'class2','test',f) for f in os.listdir(os.path.join(data_directory,'class2','test')) ]

X_test = class1_test+class2_test
Y_test = [1 for i in range(len(class1_test))] + [2 for i in range(len(class2_test))]

Nc1 = len(class1_train)
Nc2 = len(class2_train)
N = Nc1 + Nc2

def preprocess(text):
    text = text.lower()             #lower all alphabets 
    punc_table = str.maketrans({key: None for key in string.punctuation+"–"+"’"+"…"})
    text = text.translate(punc_table)
    return text

def tokenize(text):
    def isNumeric(s):
        try:
            float(s)
            return True
        except:
            return False
    text = word_tokenize(text)    #tokenize the text
    stop = set(stopwords.words('english'))            #removing stopwards
    lemma = nltk.stem.WordNetLemmatizer()           #lemmatization of word
    tokens = [lemma.lemmatize(word) for word in text if word not in stop and not isNumeric(word)]
    return tokens


class KNN:
    def __init__(self,preprocess,tokenize):
        self.vectorizer = TfidfVectorizer(input='filename',encoding='utf-8',decode_error='ignore',preprocessor=preprocess,tokenizer=tokenize,
                            norm='l2',use_idf=True,sublinear_tf=True)

    def train(self,Xcorpus,label):
        self.X = self.vectorizer.fit_transform(Xcorpus).toarray()
        self.label = label
    
    def predict(self,doc,k):
        testX = self.vectorizer.transform([doc]).toarray()
        Sk = self.nearestNeighbours(testX,k)
        scores = [0,0]
        for idx in Sk:
            score  = np.dot(testX,self.X[idx,:])[0]
            c = self.label[idx]
            scores[c-1] += score
        return scores.index(max(scores))+1  ## 1 or 2

    def nearestNeighbours(self,X,k):
        distnace = []
        for r in range(self.X.shape[0]):
            distnace.append((np.linalg.norm(self.X[r,:]-X),r))
        neighbours = nsmallest(k,distnace)
        neighbours = [ idx for d,idx in neighbours]
        return neighbours


corpus = class1_train+class2_train
label = [1 for i in range(Nc1)] + [2 for i in range(Nc2)]
knn = KNN(preprocess,tokenize)
knn.train(corpus,label)

header = "k    1    10    50"
line ="KNN"
for k in [1,10,50]:
    print("predicting test data with k = {}".format(k))
    predictions = []
    for file in X_test:
        predictions.append(knn.predict(file,k))
    
    F1 = f1_score(Y_test,predictions,average='macro')
    line += "    "+str(F1)

with open(output_file,"w") as f:
    f.write(header+'\n'+line)