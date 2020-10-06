import pickle
from trie import Trie,Node
import time
import sys

print("Inverted Index loading....")
t2 = time.time()
with open("Inverted_Index.pkl","rb") as f:
    inverted_index = pickle.load(f)
print("done!!")
print("time taken: {}".format(time.time()-t2))
print("Trie loading....")
t2 = time.time()
with open("trie.pkl","rb") as f:
    trie = pickle.load(f)
print("done!!")
print("time taken: {}".format(time.time()-t2))


def wildcard_search(query_term):
    cursor = trie.prefix_search(query_term)
    terms = trie.all_words(cursor,query_term)
    postings = {}
    for term in terms:
        term = term.split('$')
        term = term[1]+term[0]  ## rotate around $
        postings[term] = inverted_index[term]['posting']
    return postings


def query(query_term):
    parts = query_term.split('*')
    if len(parts)==1:  ## exact search
        if query_term not in inverted_index.keys():
            return {}
        else:
            return {query_term:inverted_index[query_term]['posting']}

    elif parts[0] == "":  ## of type *mon
        return  wildcard_search(parts[1]+"$")
    elif parts[1] == "":   ## of type mo*
        return wildcard_search("$"+parts[0])
    elif parts[0]!="" and parts[1]!="":   ## of type m*n
        return wildcard_search(parts[1]+"$"+parts[0])


def serialize_posting(postings):
    s = ""
    for term,posting in postings.items():
        s1=term+":"
        for pos in posting:
            s1+="<{},{}>,".format(pos[0],pos[1])
        s1=s1[:-1]+";"
        s+=s1
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
    for query_term in f:
        count+=1
        t1 = time.time()
        postings = query(query_term.strip().lower())  ## query each term
        tm += time.time()-t1
        s = serialize_posting(postings)
        of.write(s)

of.close()
print("avg time per query : ",tm/count)
