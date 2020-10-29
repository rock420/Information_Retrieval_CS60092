import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from collections import Counter
import pickle5 as pickle
import string
import os
import sys
import math

directory = "./ECTText/"
files = os.listdir(directory)
files.sort(key=lambda filename: int(filename.split(".")[0]))
inverted_index = {}
championListLocal = {}
championListGlobal = {}
N = len(files)

"""
inverted_index structure ->
{
    token1 : {
                "idf" : idf(t1),
                "posting": [
                              (docId1,tf(t1,d1)),
                              (docId2,tf(t1,d2)),
                              (docId3,tf(t1,d3)),
                              (docId4,tf(t1,d4)),
                              (docId5,tf(t1,d5)),
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
    punc_table = str.maketrans({key: None for key in string.punctuation+"–"+"’"+"…"})
    text=text.translate(punc_table)
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

def build_index(tokens,docId):
    docCount = Counter(tokens)

    for token,freq in docCount.items():
        if(token not in inverted_index.keys()):
            inverted_index[token] = {'posting':[]}
        tf = math.log10(1+freq)
        inverted_index[token]['posting'].append((docId,tf))

with open("./Data/StaticQualityScore.pkl","rb") as f:
    g = pickle.load(f)

for file in files:
    docId = int(file.split(".")[0])
    with open(directory+file,"r") as f:
        text = f.read()
    tokens = preprocess(text)
    build_index(tokens,docId)
    sys.stdout.write("\r{0}/{1} files indexed...,{2}>".format(docId+1,len(files),"="*(docId//100)))
    sys.stdout.flush()

print("\n")
for count,term in enumerate(inverted_index.keys()):
    idf = math.log10(N/len(inverted_index[term]['posting']))
    inverted_index[term]['idf'] = idf
    arr = sorted(inverted_index[term]['posting'],key=lambda x: x[1])
    championListLocal[term] = arr[:50]
    arr = sorted(inverted_index[term]['posting'],key= lambda x: g[x[0]]+(x[1]*idf))
    championListGlobal[term] = arr[:50]
    sys.stdout.write("\r{0}/{1} terms processed....".format(count+1,len(inverted_index)))
    sys.stdout.flush()


with open("Inverted_Index.pkl","wb") as f:
    pickle.dump(inverted_index,f)

with open("ChampionListLocal.pkl","wb") as f:
    pickle.dump(championListLocal,f)

with open("ChampionListGlobal.pkl","wb") as f:
    pickle.dump(championListGlobal,f)
