import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from collections import Counter
import string
import os
import sys
import math
import numpy as np


data_directory = "./dataset/"
class1_train = [os.path.join(data_directory,'class1','train',f) for f in os.listdir(os.path.join(data_directory,'class1','train')) ]
class2_train = [os.path.join(data_directory,'class2','train',f) for f in os.listdir(os.path.join(data_directory,'class2','train')) ]
class1_test =  [os.path.join(data_directory,'class1','test',f) for f in os.listdir(os.path.join(data_directory,'class1','test')) ]
class2_test =  [os.path.join(data_directory,'class2','test',f) for f in os.listdir(os.path.join(data_directory,'class2','test')) ]


Nc1 = len(class1_train)
Nc2 = len(class2_train)
N = Nc1 + Nc2
vocab = {}
docCount = {}
total_count = [0,0]
"""
    vocab ->
    {
        term1 : [ #conditioanl prob for class1, #conditional prob for class2 ],
        term2 : [ #conditioanl prob for class1, #conditional prob for class2 ],
        .
        .
        .
    }
"""

def preprocess(text):
    text = text.lower()             #lower all alphabets 
    punc_table = str.maketrans({key: None for key in string.punctuation+"–"+"’"+"…"})
    text = text.translate(punc_table)
    text = word_tokenize(text)    #tokenize the text
    stop = set(stopwords.words('english'))            #removing stopwards
    lemma = nltk.stem.WordNetLemmatizer()           #lemmatization of word
    tokens = [lemma.lemmatize(word) for word in text if word not in stop and not isNumeric(word)]
    return tokens

def isNumeric(s):
    try:
        float(s)
        return True
    except:
        return False

def build_vocab(tokens,c):
    tokensCount = Counter(tokens)
    for term,doc_freq in tokensCount.items():  ## terms are unique here
        if term not in vocab.keys():
            vocab[term] = [0,0]
        vocab[term][c-1] += doc_freq
        total_count[c-1] += doc_freq

        if term not in docCount.keys():
            docCount[term] = [0,0]
        docCount[term][c-1] += 1


def selectFeatures(docCount,Nc1,Nc2,k):
    L = []
    for term in docCount.keys():
        A = computeMI(docCount,term,Nc1,Nc2)
        L.append((A,term))

    L.sort(reverse=True)
    Features = set([term for A,term in L[:k]])
    return Features

def computeMI(docCount,term,Nc1,Nc2):
    N11 = docCount[term][0]
    N10 = docCount[term][1]
    N01 = Nc1 - docCount[term][0]
    N00 = Nc2 - docCount[term][1]
    N = Nc1 + Nc2

    A = 0
    if N11 !=0:
        A += (N11/N)*math.log2((N*N11)/((N11+N10)*(N11+N01))) 
    if N01 !=0:
        A += (N01/N)*math.log2((N*N01)/((N01+N00)*(N11+N01))) 
    if N10 !=0:
        A += (N10/N)*math.log2((N*N10)/((N11+N10)*(N10+N00))) 
    if N00 !=0:
        A += (N00/N)*math.log2((N*N00)/((N01+N00)*(N10+N00)))

    return A

def F1_Score(class1_predictions,class2_predictions):
    c1_pred_count = Counter(class1_predictions)
    c2_pred_count = Counter(class2_predictions)
    ## class 1 -> positive, class 2->negative
    TP = c1_pred_count[1]
    FN = c1_pred_count[2]
    FP = c2_pred_count[1]
    TN = c2_pred_count[2]
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = (2*Precision*Recall)/(Precision+Recall)
    return F1

class MultiNomial_NB:

    def __init__(self,):
        self.conditioanl = {}
        self.prior = None
        self.B = 0
        self.total_count = [0,0]

    def train(self,vocab,total_count,Nc1,Nc2,Features=None):
        N = Nc1 + Nc2
        self.prior = (Nc1/N,Nc2/N)  ## prior probabilities

        self.B = len(vocab)  ## Size of the vocabulary
        self.total_count = total_count
        if Features == None:
            Features = set(vocab.keys())

        for term in vocab.keys():
            self.conditioanl[term] = [0,0]
            if term in Features:
                self.conditioanl[term][0] = (vocab[term][0]+1)/(total_count[0]+self.B)  ## class 1
                self.conditioanl[term][1] = (vocab[term][1]+1)/(total_count[1]+self.B)  ## class 2
            else:
                self.conditioanl[term] = [1/(total_count[0]+self.B),1/(total_count[1]+self.B)]

    def predict(self,doc):
        with open(doc,'r',encoding="utf8", errors='ignore') as f:
           text = f.read()
        tokens = preprocess(text)
        scores = [math.log(self.prior[0]),math.log(self.prior[1])]
        
        for token in tokens:
            if token in self.conditioanl.keys():
                scores[0] += math.log(self.conditioanl[token][0])
                scores[1] += math.log(self.conditioanl[token][1])
            else:
                scores[0] += math.log(1/(self.total_count[0]+self.B))
                scores[1] += math.log(1/(self.total_count[1]+self.B))
        return scores.index(max(scores))+1  ## 1 or 2



class Bernoulli_NB:

    def __init__(self):
        self.conditioanl = {}
        self.prior = None
        self.B = 2

    def train(self,docCount,Nc1,Nc2,Features=None):
        N = Nc1 + Nc2
        self.prior = (Nc1/N,Nc2/N)  ## prior probabilities

        if Features == None:
            Features = set(docCount.keys())
        for term in docCount.keys():
            if term in Features:
                self.conditioanl[term] = [0,0]
                self.conditioanl[term][0] = (docCount[term][0]+1)/(Nc1+self.B)  ## class 1
                self.conditioanl[term][1] = (docCount[term][1]+1)/(Nc2+self.B)  ## class 2
            else:
                self.conditioanl[term] = [1/(Nc1+self.B),1/(Nc2+self.B)]
        
    
    def predict(self,doc):
        with open(doc,'r',encoding="utf8", errors='ignore') as f:
           text = f.read()
        tokens = preprocess(text)
        scores = [math.log(self.prior[0]),math.log(self.prior[1])]
        
        for token in self.conditioanl.keys():
            if token in tokens:
                scores[0] += math.log(self.conditioanl[token][0])
                scores[1] += math.log(self.conditioanl[token][1])
            else:
                scores[0] += math.log(1-self.conditioanl[token][0])
                scores[1] += math.log(1-self.conditioanl[token][1])
        return scores.index(max(scores))+1  ## 1 or 2


print("Building Vocab...")
for file in class1_train:
    with open(file,"r",encoding="utf8", errors='ignore') as f:
        text = f.read()
    tokens = preprocess(text)
    build_vocab(tokens,1)

for file in class2_train:
    with open(file,"r",encoding="utf8", errors='ignore') as f:
        text = f.read()
    tokens = preprocess(text)
    build_vocab(tokens,2)
print("done")


header = "NumFeature    1    10    100    1000    10000"
line1 ="MultinomialNB"
line2 = "BernoulliNB" 
for x in [1,10,100,1000,10000]:
    print("Naive Bayes with Feature selection, x = {}".format(x))
    Features = selectFeatures(docCount,Nc1,Nc2,x)
    mNB = MultiNomial_NB()
    mNB.train(vocab,total_count,Nc1,Nc2,Features)
    bNB = Bernoulli_NB()
    bNB.train(docCount,Nc1,Nc2,Features)
    mNB_class1_predictions = []
    mNB_class2_predictions = []
    bNB_class1_predictions = []
    bNB_class2_predictions = []
    print("testing....")
    for file in class1_test:
        mNB_class1_predictions.append(mNB.predict(file))
        bNB_class1_predictions.append(bNB.predict(file))
    for file in class2_test:
        mNB_class2_predictions.append(mNB.predict(file))
        bNB_class2_predictions.append(bNB.predict(file))
    print("done...")
    mNB_F1 = F1_Score(mNB_class1_predictions,mNB_class2_predictions)
    bNB_F1 = F1_Score(bNB_class1_predictions,bNB_class2_predictions)
    line1 += "    "+str(mNB_F1)
    line2 += "     "+str(bNB_F1)

of = open("result.txt","w")
of.write(header+'\n'+line1+'\n'+line2)