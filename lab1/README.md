### USAGE INFO ###

It is required to have the following (empty) directories in /lab1

a) lab1/corpus

b) lab1/data

c) lab1/fsts

d) lab1/vocab

Required files:

a) data/ab.syms (for miniL,V,S fsts)

b) data/wiki.txt

Then, execute the files of lab1/scripts directory in the following order:

1) fetch_gutenberg.py (creates corpus/gutenberg.txt)
2) find_edits.py (requires data/wiki.txt, creates word_edits.txt)
3) create_dict_symbols.py (creates vocab/words.syms, vocab/chars.syms, data/word_edits.txt)
4) create_fsts (create fsts/L,V,S,M,N,E,W.fst)
5) fst_commands.sh (compiles,composes etc all necessary fsts and fstdraw images)


NOTE: data/aclImdb is not included
