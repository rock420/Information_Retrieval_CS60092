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
    Score.sort(reverse=True)
    return Score[:10]

def local_Champion_List_Score(Q):
    Score = {}
    Q_norm = 0
    for term in Q:
        if term in championListLocal.keys():
            idf = inverted_index[term]['idf']
            Q_norm += idf**2
            for docId,tf in championListLocal[term]:
                if docId not in Score.keys():
                    Score[docId] = 0
                Score[docId] += idf*(tf*idf)
                
    if(Q_norm==0):
        return []
    Q_norm = math.sqrt(Q_norm)
    score_list = []
    for docId in Score.keys():
        score_list.append((Score[docId]/(Q_norm*doc_norm[docId]),docId))
    
    ## return top 10 docs
    score_list.sort(reverse=True)
    return score_list[:10]

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
t2 = t4 = 0
print("start Query..",flush=True)
with open(query_file,"r") as f:
    for query in f:
        if query[-1]=="\n":
            query = query[:-1]
        of.write(query+"\n")
        count+=1
        Q = preprocess_Query(query)

        t1 = time.time()
        docScore= tfidf_Score(Q)
        t2 += (time.time()-t1)
        print(docScore)
        s = serialize_output(docScore)
        of.write(s)

        t3 = time.time()
        docScore = local_Champion_List_Score(Q)
        t4 += (time.time()-t3)
        print(docScore)
        s = serialize_output(docScore)
        of.write(s)

of.close()
print("avg time per query using inverted index: ",t2/count)
print("avg time per query using local champiolist: ",t4/count)