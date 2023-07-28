from helpers import read_wiki_txt
from helpers import run_cmd
from math import log

location = "../data/wiki.txt"
pairs = read_wiki_txt(location)   #pair of misspelled and corecr word

## save all found edits to the following location
location = "../data/word_edits.txt"
file = open(location,"w")

for pair in pairs:
    edits = run_cmd(f"./word_edits.sh {pair[0]} {pair[1]}")
    file.write(edits)
file.close()


