
class Node():
    def __init__(self,key,leaf=False):
        self.key = key   ## key=None for root
        self.childs = []
        self.leaf = leaf
        self.endOfWord = False


class Trie():
    def __init__(self):
        self.root = None

    def insert(self,term):

        
        if self.root == None:
            self.root = Node(None,True)
        
        i = 0
        cursor = self.root
        prev = self.root
        while i<len(term) and not cursor.leaf:
            for child in cursor.childs:
                if child.key==term[i]:
                    cursor = child
                    break
            if prev==cursor:
                break
            else: prev = cursor
            i+=1

        while i<len(term):
            newNode = Node(term[i],True)
            cursor.leaf = False
            cursor.childs.append(newNode)
            cursor = newNode
            i+=1

        cursor.endOfWord = True

    def prefix_search(self,prefix):
        if self.root==None:
            return None

        cursor = self.root
        prev = self.root
        for ch in prefix:
            if cursor.leaf:
                return None
            for child in cursor.childs:
                if child.key == ch:
                    cursor = child
                    break
            if prev == cursor:
                return None
            else: prev = cursor

        return cursor
            
    
    def all_words(self,cursor,prefix):
        if cursor==None:
            return []
        if cursor.leaf:
            return [prefix]    

        terms = []
        if cursor.endOfWord:
            terms.append(prefix)

        for child in cursor.childs:
            tmp = self.all_words(child,prefix+child.key)
            terms = terms+tmp

        return terms


if __name__ == "__main__":
    arr = ["a", "aardvark", "hu", "huygens", "m", "si", "sickle", "z", "zygot","sicu"]
    tr = Trie()
    for x in arr:
        tr.insert(x)

    ## prefix search
    prefix = "sic"
    m = tr.prefix_search(prefix)
    words = tr.all_words(m,prefix)
    print(words)

    ## exact search
    search = "sic"  
    m = tr.prefix_search(search)
    if m.endOfWord:
        print("found")
    else:
        print("not present")
   