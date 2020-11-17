import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
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

def F1_Score(class1_predictions,class2_predictions):
    c1_pred_count = Counter(class1_predictions)
    c2_pred_count = Counter(class2_predictions)
    ## class 1 -> positive, class 2->negative
    TP = c1_pred_count[1]
    FN = c1_pred_count[2]
    FP = c2_pred_count[1]
    TN = c2_pred_count[2]
    print(TP," ",FP," ",FN," ",TN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = (2*Precision*Recall)/(Precision+Recall)
    return F1

class Rocchio:
    def __init__(self,preprocess,tokenize):
        self.vectorizer = TfidfVectorizer(input='filename',encoding='utf-8',decode_error='ignore',preprocessor=preprocess,tokenizer=tokenize,
                            norm='l2',use_idf=True,sublinear_tf=True)
        self.u = []

    def train(self,Xcorpus,Nc1,Nc2):
        X = self.vectorizer.fit_transform(Xcorpus).toarray()
        X1 = X[0:Nc1,:]
        X2 = X[Nc1:,:]
        u1 = np.mean(X1,axis=0)
        u2 = np.mean(X2,axis=0)
        self.u = [u1,u2]

    def predict(self,doc,b):
        X = self.vectorizer.transform([doc]).toarray()
        d1 = np.linalg.norm(self.u[0]-X)
        d2 = np.linalg.norm(self.u[1]-X)

        if d1<d2-b:
            return 1
        else:
            return 2



corpus = class1_train+class2_train
print("training...")
rochhio = Rocchio(preprocess,tokenize)
rochhio.train(corpus,Nc1,Nc2)
print("done")

header = "b    0    .01    .05    .1"
line ="Rocchio"
for b in [0,0.01,0.05,0.1]:
    print("predicting test data with b = {}".format(b))
    class1_predictions = []
    class2_predictions = []
    for file in class1_test:
        class1_predictions.append(rochhio.predict(file,b))
    for file in class2_test:
        class2_predictions.append(rochhio.predict(file,b))
    
    F1 = F1_Score(class1_predictions,class2_predictions)
    line += "    "+str(F1)

of = open("result1.txt","w")
of.write(header+'\n'+line)