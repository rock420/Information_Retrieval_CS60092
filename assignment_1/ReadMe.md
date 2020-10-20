# Building Inverted Positional Index and Answering Specialized Wildcard Queries

This assignment is on creating text corpus from html form data, building inverted positional index on that dataset and using them to answer wildcard queries.

**scrap_data.py** -> This file scrap data from "https://seekingalpha.com" and store each file under ECT folder.

**build_corpus.py** -> This file first build the ECTNestedDict and then generate text corpus from corresponding dictionary.ECTNestedDict is save as "ECTNestedDict.pkl" and the text extracted from each file is save in corresponding file in "ECTText" folder. It also generate "FileMap.pkl" which contain mapping from docId to corresponding filename.

**build_index.py** -> This file first preprocess the text and then generate the Inverted Index. It also build a Trie using `permuterm index` for prefix search. It load each file saved under "ECTText" to generate the Inverted Index. This file save the index in "Inverted_Index.pkl" and the Trie of permuterm indexes in "trie.pkl".

**retrieve.py** -> It first load the "Inverted_Index.pkl" and "trie.pkl" file from the directory. Then it start query by reading each term in "query.txt" line by line and save the corresponding result in the require format(by serializing the posting) in "result.txt" file.


### Require Python libraries: (python 3)
    1. BeautifulSoup
    2. nltk
    3. pickle
    4. re
    5. requests