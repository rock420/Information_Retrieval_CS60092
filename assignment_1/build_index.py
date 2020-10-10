import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import pickle
import string
import os
import sys
from trie import Trie

directory = "./ECTText/"
files = os.listdir(directory)
files.sort(key=lambda filename: int(filename.split(".")[0]))
inverted_index = {}
vocab = set()

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
    punc_table = str.maketrans({key: None for key in string.punctuation+"–"+"’"})
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
    terms = set()
    for pos,token in enumerate(tokens):
        if token in inverted_index:
            if(token not in terms): inverted_index[token]['docFreq']+=1
            inverted_index[token]['posting'].append((docId,pos))
        
        else:
            inverted_index[token] = {'docFreq':1,'posting':[(docId,pos)]}
        terms.add(token)
        vocab.add(token)


for file in files:
    docId = int(file.split(".")[0])
    with open(directory+file,"r") as f:
        text = f.read()
    tokens = preprocess(text)
    build_index(tokens,docId)
    sys.stdout.write("\r{0}/{1} files indexed...,{2}>".format(docId+1,len(files),"="*(docId//100)))
    sys.stdout.flush()

with open("Inverted_Index.pkl","wb") as f:
    pickle.dump(inverted_index,f)


trie = Trie()
print("\n")
for t,term in enumerate(vocab):
    dkey = term+'$'
    for i in range(len(dkey),0,-1):
        permuterm = dkey[i:]+dkey[:i]
        trie.insert(permuterm)
    sys.stdout.write("\r{0}/{1} terms processed.....".format(t+1,len(vocab)))
    sys.stdout.flush()

with open("trie.pkl","wb") as f:
    pickle.dump(trie,f)
