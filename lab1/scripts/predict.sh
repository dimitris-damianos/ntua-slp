#!/usr/bin/env bash

# Run spell corrector for an input word

# Usage:
#   bash scripts/predict.sh MY_SPELL_CHECKER tst
# Output:
#   test

# Command line args
SPELL_CHECKER_COMPILED=${1}
WORD=${2}


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

###
# Make sure these files exist
CHARSYMS=${CURRENT_DIRECTORY}/../vocab/chars.syms
WORDSYMS=${CURRENT_DIRECTORY}/../vocab/words.syms
###

## NOTE: if predict.sh is called from lab1 -> change to scripts/mkfstinput.py

# Make input fst for the misspelled word
python3 mkfstinput.py ${WORD} |
    # Compile and compose with the spell checker
    fstcompile --isymbols=${CHARSYMS} --osymbols=${CHARSYMS} |
    fstcompose - ${SPELL_CHECKER_COMPILED} |
    # Get shortest path and sort arcs
    fstshortestpath |
    fstrmepsilon |
    fsttopsort |
    # print output fst using words.syms
    fstprint -osymbols=${WORDSYMS} |
    # Get destination word (corrected)
    cut -f4 |
    # Ignore epsilon outputs
    grep -v "<epsilon>" |
    # Ignore accepting state line
    head -n -1 |
    # Remove trailing new line
    tr -d '\n'