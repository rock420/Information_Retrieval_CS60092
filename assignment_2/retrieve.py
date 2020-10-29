import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle5 as pickle
import time
import math
import sys
import string 

with open("Inverted_Index.pkl","rb") as f:
    inverted_index = pickle.load(f)

with open("ChampionListLocal.pkl","rb") as f:
    championListLocal = pickle.load(f)

with open("ChampionListGlobal.pkl","rb") as f:
    championListGlobal = pickle.load(f)

N = 1000
doc_norm = [0 for i in range(N)]
for term in inverted_index.keys():
    idf = inverted_index[term]['idf']
    for  docId,tf in inverted_index[term]['posting']:
        doc_norm[docId] += (tf*idf)**2
for i in range(N):
    doc_norm[i] = math.sqrt(doc_norm[i])


def preprocess_Query(text):
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

def tfidf_Score(Q):
    Score = [0 for i in range(N)]
    Q_norm = 0
    for term in Q:
        if term in inverted_index.keys():
            idf = inverted_index[term]['idf']
            Q_norm += idf**2
            for docId,tf in inverted_index[term]['posting']:
                Score[docId] += idf*(tf*idf)
    Q_norm = math.sqrt(Q_norm)
    for i in range(N):
        Score[i] = (Score[i]/(Q_norm*doc_norm[i]),i)
    Score.sort()
    return Score[:10]


def serialize_output(doc_score):
    s = ""
    for score,docId in doc_score:
        s += "< {},{} >,".format(docId,score)
    s = s[:-1]+"\n"
    return s


n = len(sys.argv)
if n<2:
    query_file = "query.txt"
else:
    query_file = sys.argv[1]

of = open("result.txt","w")
count = 0
tm = 0
print("start Query..",flush=True)
with open(query_file,"r") as f:
    for query in f:
        if query[-1]=="\n":
            query = query[:-1]
        of.write(query+"\n")
        count+=1
        t1 = time.time()
        Q = preprocess_Query(query)
        doc_score= tfidf_Score(Q)
        print(doc_score)
        tm += time.time()-t1
        s = serialize_output(doc_score)
        of.write(s)

of.close()
print("avg time per query : ",tm/count)