import re
from bs4 import BeautifulSoup
import os
import pickle
import sys

directory = "./ECT/"
files = os.listdir(directory)
dict_corpus = {}
filename_map = {}

months = ["January","February","March","April","May","June","July","August","September","October","November","December",""]


def create_dictionary(filename):

    with open(filename,'r') as f:
        html_doc = f.read()
    soup = BeautifulSoup(html_doc,'html.parser')

    p_list = soup.find_all('p')
    date,idx = get_date(p_list)
    p_list = p_list[idx:]
    participants,idx = get_participants(p_list)
    p_list = p_list[idx:]
    presentation,idx = get_presentation_dictionary(p_list)
    p_list = p_list[idx:]
    questionnaire = get_questionnaire_dictionary(p_list)

    transcript = {"Date":date,"Participants":participants,"Presentation":presentation,"Questionnaire":questionnaire}
    return transcript


def get_date(p_list):
    regex = r"\s*\d{1,2},{0,1}\s*\d{4}|".join(months)[:-1]
    idx = 0
    for para in p_list:
        if para.text=="Company Participants":
            break
        idx+=1
        match = re.findall(regex,para.text)
        if len(match)!=0:
            break
    return match[0],idx

def get_participants(p_list):

    participants=[]
    count = 0
    idx = -1
    for para in p_list:
        if count==3: 
            break
        idx+=1
        if para.strong!=None:
            count+=1
            continue
        name = para.text
        participants.append(name)
        
    return participants,idx

def get_presentation_dictionary(p_list):
    presentation = {}
    name = ""
    value = ""
    idx = -1
    for para in p_list:
        idx +=1
        if para.strong!=None:
            if name not in presentation.keys():
                presentation[name] = []
            presentation[name].append(value)
            if para.has_attr('id') and para['id']=="question-answer-session":
                break
            name = para.text
            value = ""
            continue
        value += para.text+" "

    presentation.pop("")
    return presentation,idx

def get_questionnaire_dictionary(p_list):
    questionnaire = {}
    serial = 0
    name = ""
    statement = ""
    for para in p_list[1:]:
        if para.strong!=None:
            questionnaire[serial]={"Speaker":name,"Remark":statement}
            serial+=1
            name = para.text.split(" - ")[-1]
            statement = ""
            continue
        statement += para.text+" "

    questionnaire[serial]={"Speaker":name,"Remark":statement}
    questionnaire.pop(0)
    return questionnaire

"""def print_dictionary(transcript):
    print("Date: {}".format(transcript['Date']))
    print("Participants: ",transcript["Participants"])
    print("\nPresentation\n")
    for presenter,statements in transcript["Presentation"].items():
        print("{} :".format(presenter))
        for statement in statements:
            print(statement)
        print("\n")
    print("\nQuestion-Answer :{}\n".format(len(transcript["Questionnaire"])))
    for key,value in transcript["Questionnaire"].items():
        print("{} :".format(value["Speaker"]))
        print(value["Remark"])
        print("\n")"""

def build_text(dict_transcript):

    text = ""
    text += "Date"+" "+dict_transcript["Date"]+" \n"
    text += "Participants \n"
    for participant in dict_transcript["Participants"]:
        text+=participant+" \n"
    text += "Presentation \n"
    for presenter,statements in dict_transcript["Presentation"].items():
        text += presenter+" \n"
        for statement in statements:
            text += statement+" \n"
    text += "Questionnaire"+" \n"
    for serial,statement in dict_transcript["Questionnaire"].items():
        text += str(serial)+" "+statement["Speaker"]+" \n"
        text += statement["Remark"]+" \n"

    return text



docId = 0
if(not os.path.isdir("./ECTText/")):
    os.mkdir("./ECTText/")

for filename in files:
    try:
        transcript = create_dictionary(directory+filename)
    except Exception as e:
        #print(filename)
        #print(e)
        continue
    filename_map[docId] = filename
    dict_corpus[docId] = transcript
    text = build_text(transcript)
    with open("./ECTText/"+str(docId)+".txt","w") as f:
        f.write(text)
    docId+=1
    sys.stdout.write("\r{0}/{1} files processed...,{2}>".format(docId,len(files),"="*(docId//100)))
    sys.stdout.flush()

with open("ECTNestedDict.pkl","wb") as f:
    pickle.dump(dict_corpus,f)

with open("FileMap.pkl","wb") as f:
    pickle.dump(filename_map,f)