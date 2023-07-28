### DICTIONARY 

## corpus file location
filename = "../corpus/gutenberg.txt"

dict = {}

## Opening file
with open(filename,"r") as file:
    for line in file:
        for word in line.split():
            if word not in dict.keys():
                dict[word] = 1
            else:
                dict[word]+=1

## filtering: 
## delete those words that appear less than 5 times
deleted = []
for key in dict.keys():
    if dict[key]<5:
        deleted.append(key)
for k in deleted:
    del dict[str(k)]
   
## the existence of vocab folder is required     
## write tokens and frequency to vocab/words.vocab.txt
location = "../vocab/words.vocab.txt"
f= open(location,"w")

for key in dict.keys():
    f.write(str(key)+"\t"+str(dict[str(key)])+"\n")
f.close()

# -------------------------------------------------------------------- #

### INPUT/OUTPUT SYMBOLS

## Create chars.syms
alphabet = ['<epsilon>','a','b','c','d','e','f','g','h','i','j',
            'k','l','m','n','o','p','q','r','s','t',
            'u','v','w','x','y','z']
location = "../vocab/chars.syms"
f = open(location,"w")
index = 0
for c in alphabet:
    f.write(str(c) +"\t"+str(index)+"\n")
    index+=1
f.close()

## Create words.syms
location = "../vocab/words.syms"
f = open(location,"w")
index = 0
f.write("<epsilon>"+"\t"+str(index)+"\n")
index+=1
for word in sorted(dict.keys()): #print words alphabetically
    f.write(str(word)+"\t"+str(index)+"\n")
    index+=1
f.close()

## --------------------------------------------------------------------------------- ##
### FIND ALL EDITS FROM wiki.txt

from helpers import read_wiki_txt
from helpers import run_cmd
from math import log


location = "../data/wiki.txt"
pairs = read_wiki_txt(location)   #pair of misspelled and correct word


## save all found edits to the following location
location = "../data/word_edits.txt"
file = open(location,"w")

for num,pair in enumerate(pairs):
    edits = run_cmd(f"./word_edits.sh {pair[0]} {pair[1]}")
    file.write(edits)
    if num%10 == 0:
        print(f"pair num -> {num}")
file.close()
