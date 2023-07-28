'''
## LEVENSHTEIN TRANSDUCER L.fst
## Alphabet of our FST
alphabet = ['a','b','c','d','e','f','g','h','i','j',
            'k','l','m','n','o','p','q','r','s','t',
            'u','v','w','x','y','z']

## Weights for the chosen transition
weights = {"no edit":0.0,
           "deletion":1.0,
           "insertion":1.0,
           "to other":1.0}

## create L.fst file
location = "../fsts/L.fst"
file = open(location,"w")
## no edit
for letter in alphabet:
    file.write(str(0)+"\t"+str(0)+"\t"+str(letter)+"\t"+str(letter)+"\t"+str(weights["no edit"]))
    file.write("\n")
## deletion 
for letter in alphabet:
    file.write(str(0)+"\t"+str(0)+"\t"+str(letter)+"\t"+"<epsilon>"+"\t"+str(weights["deletion"]))
    file.write("\n")
## insertion
for letter in alphabet:
    file.write(str(0)+"\t"+str(0)+"\t"+"<epsilon>"+"\t"+str(letter)+"\t"+str(weights["insertion"]))
    file.write("\n")   
## to other character
for letter in alphabet:
    for l in alphabet:
        if letter != l:
            file.write(str(0)+"\t"+str(0)+"\t"+str(letter)+"\t"+str(l)+"\t"+str(weights["to other"]))
            file.write("\n")
## add final 
file.write(str(0)+"\t"+str(0.0)+"\n")

file.close()

## Repeat for a mini_L.fst 
alphabet2 = "ab"

location = "../fsts/miniL.fst"
file = open(location,"w")

## no edit
for letter in alphabet2:
    file.write(str(0)+"\t"+str(0)+"\t"+str(letter)+"\t"+str(letter)+"\t"+str(weights["no edit"])+"\n")
## deletion 
for letter in alphabet2:
    file.write(str(0)+"\t"+str(0)+"\t"+str(letter)+"\t"+"<epsilon>"+"\t"+str(weights["deletion"])+"\n")
## insertion
for letter in alphabet2:
    file.write(str(0)+"\t"+str(0)+"\t"+"<epsilon>"+"\t"+str(letter)+"\t"+str(weights["insertion"])+"\n")    
## to other character
for letter in alphabet2:
    for l in alphabet2:
        if letter != l:
                file.write(str(0)+"\t"+str(0)+"\t"+str(letter)+"\t"+str(l)+"\t"+str(weights["to other"])+"\n")

## add final 
file.write(str(0)+"\t"+str(0.0)+"\n")

file.close()

### ACCEPTOR V.fst
## open and read vocab/words.syms
location = "../vocab/words.syms"
file = open(location,"r")
lines = file.readlines()
words = []                         
for line in lines:                  # words will be saved in "words" list
    words.append(line.split()[0])   # add to words the word from each line
file.close()

## create an acceptor for the 4 first words (minV.fst)
cost = 0.0 
location = "../fsts/miniV.fst"
file = open(location,"w")
demo = [words[1],words[2],words[3],words[4]] #ignore 1st word since its <epsilon>
state = 1
for w in demo:
    index = 0
    file.write(str(0)+"\t"+str(state)+"\t"+str(w[0])+"\t"+str(w)+"\t"+str(cost)+"\n")
    index+=1
    state+=1
    while index<len(w):
        file.write(str(state-1)+"\t"+str(state)+"\t"+str(w[index])+"\t"+"<epsilon>"+"\t"+str(cost)+"\n")
        index+=1
        state+=1
    file.write(str(state-1)+"\t"+str(cost)+"\n")

## create an acceptor for the all words (V.txt)
cost = 0.0 
location = "../fsts/V.fst"
file = open(location,"w")
state = 1
for w in words[1:]:
    index = 0
    file.write(str(0)+"\t"+str(state)+"\t"+str(w[0])+"\t"+str(w)+"\t"+str(cost)+"\n")
    index+=1
    state+=1
    while index<len(w):
        file.write(str(state-1)+"\t"+str(state)+"\t"+str(w[index])+"\t"+"<epsilon>"+"\t"+str(cost)+"\n")
        index+=1
        state+=1
    file.write(str(state-1)+"\t"+str(cost)+"\n")
    
## -------------------------------------------------------------------------------------------------------------- ##   

### CREATE MLN.fst

### M.fst
## Read from data/wiki.txt file
location = "../data/wiki.txt"
file = open(location,"r")
lines = file.readlines()

misspelled = [] #contains the 1st column of wiki.txt
correct = []    #contains the 2nd column of wiki.txt

for line in lines:
    misspelled.append(line.split()[0])
    correct.append(line.split()[1])
file.close()


## create acceptor M
location = "../fsts/M.fst"
file = open(location,"w")

word = misspelled[0]
state = 1
index = 0
file.write(str(0)+"\t"+str(state)+"\t"+str(word[index])+"\t"+str(word[index])+"\n")
index+=1
state+=1
while index<len(word):
    file.write(str(state-1)+"\t"+str(state)+"\t"+str(word[index])+"\t"+str(word[index])+"\n")
    index+=1
    state+=1
    
file.write(str(state-1)+"\n")
file.close()

## create transducer N
location = "../fsts/N.fst"
file = open(location,"w")

word = correct[0]
state = 1
index = 0
file.write(str(0)+"\t"+str(state)+"\t"+str(word[index])+"\t"+str(word[index])+"\n")
index+=1
state+=1
while index<len(word):
    file.write(str(state-1)+"\t"+str(state)+"\t"+str(word[index])+"\t"+str(word[index])+"\n")
    index+=1
    state+=1
    
file.write(str(state-1)+"\n")
file.close()

'''       
## --------------------------------------------------------------------------------------------- ##
      
## CREATE E.fst EDIT DISTANCE TRANSDUCER
## find the frequency of each edit from wiki.txt
frequency = {}
location = "../data/word_edits.txt"
file = open(location,"r")

lines = file.readlines()
for line in lines:
    source = line.split()[0]
    target = line.split()[1]
    if str((source,target)) not in frequency.keys():
        frequency[str((source,target))] = 1
    else:
        frequency[str((source,target))] += 1
   

    
## create E.fst
location = "../fsts/E.fst"
file = open(location,"w")

alphabet = ['<epsilon>','a','b','c','d','e','f','g','h','i','j',
            'k','l','m','n','o','p','q','r','s','t',
            'u','v','w','x','y','z']

INF = 10000000000

## create all possible tuples of letters
## if the touple does not exist in frequency then cost <= INF
## else cost <= -log(frequency)
from math import log10

total_edits = sum(frequency.values()) #number of all edits
edits = len(frequency.values())

for let1 in alphabet:
    for let2 in alphabet: 
        if let1 == let2 and let1!='<epsilon>':          #from char to itself has 0 cost
            file.write(str(0)+"\t"+str(0)+"\t"+str(let1)+"\t"+str(let2)+"\t"+str(0.0)+"\n") 
        else:
            key = str((let1,let2)) 
            if key in frequency.keys():
                file.write(str(0)+"\t"+str(0)+"\t"+str(let1)+"\t"+str(let2)+"\t"+str(-log10(frequency[key]/total_edits))+"\n")
            else:
                file.write(str(0)+"\t"+str(0)+"\t"+str(let1)+"\t"+str(let2)+"\t"+str(INF)+"\n")
file.write(str(0))
file.close()

## ----------------------------------------------------------------------------------------------------------- ##

### W.fst
from math import log10

location = "../vocab/words.vocab.txt"
file = open(location,"r")

loc = "../fsts/W.fst"
f = open(loc,"w")

lines = file.readlines()

total_words = 0         #number of all words in gutenberg.txt
for line in lines:
    num = line.split()[1]
    total_words+=int(num)
    
for line in lines:
    word = line.split()[0]
    cost = -log10(int(line.split()[1])/total_words)
    f.write(str(0)+"\t"+str(0)+"\t"+str(word)+"\t"+str(word)+"\t"+str(cost)+"\n")
f.write(str(0))
file.close()
f.close()


## ---------------------------------------------------------------------------------------------------------- ##

## E_laplace.fst

location = "../fsts/E_laplace.fst"
file = open(location,"w")

for let1 in alphabet:
    for let2 in alphabet: 
        if let1 == let2 and let1!='<epsilon>':          #from char to itself has 0 cost
            file.write(str(0)+"\t"+str(0)+"\t"+str(let1)+"\t"+str(let2)+"\t"+str(0.0)+"\n") 
        else:
            key = str((let1,let2)) 
            if key in frequency.keys():
                file.write(str(0)+"\t"+str(0)+"\t"+str(let1)+"\t"+str(let2)+"\t"+str(-log10(frequency[key]/total_edits))+"\n")
            else:
                cost = -log10(1/(total_edits+edits))
                file.write(str(0)+"\t"+str(0)+"\t"+str(let1)+"\t"+str(let2)+"\t"+str(cost)+"\n")
file.write(str(0))
file.close()




