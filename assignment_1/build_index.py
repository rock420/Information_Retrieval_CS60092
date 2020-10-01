import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import pickle
import string
import os
import sys

directory = "./ECTText/"
files = os.listdir(directory)
files.sort(key=lambda filename: int(filename.split(".")[0]))
inverted_index = {}

"""
inverted_index structure ->
{
    token1 : {
                "docFreq" : ##document freq,
                "posting": [
                              (docId1,pos1),
                              (docId1,pos2),
                              (docId2,pos1),
                              (docId2,pos2),
                              (docId2,pos3),
                              .
                              .
                              .
                              .
                           ]
            },
    token2 : {
                .
                .
                .
    }

    .
    .
    .
    .
}
"""


def preprocess(text):
    text = text.lower()             #lower all alphabets 
    text = word_tokenize(text)    #tokenize the text
    stop = set(stopwords.words('english'))            #removing stopwards
    lemma = nltk.stem.WordNetLemmatizer()           #lemmatization of word
    tokens = [lemma.lemmatize(word) for word in text if word not in string.punctuation+"–"+"’" and word not in stop ]
    return tokens

def build_index(tokens,docId):
    terms = set()
    for pos,token in enumerate(tokens):
        if token in inverted_index:
            if(token not in terms): inverted_index[token]['docFreq']+=1
            inverted_index[token]['posting'].append((docId,pos))
        
        else:
            inverted_index[token] = {'docFreq':1,'posting':[(docId,pos)]}
        terms.add(token)



for file in files:
    docId = int(file.split(".")[0])
    with open(directory+file,"r") as f:
        text = f.read()
    tokens = preprocess(text)
    build_index(tokens,docId)
    sys.stdout.write("\r{0}/{1} files processed...,{2}>".format(docId,len(files),"="*(docId//100)))
    sys.stdout.flush()
    """if(docId%100==0):
        print("{} files processed....".format(docId))"""

with open("Inverted_Index.pkl","wb") as f:
    pickle.dump(inverted_index,f)
