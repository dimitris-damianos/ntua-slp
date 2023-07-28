# bash file for easy compile, print and draw

#compile miniL.fst
fstcompile -isymbols=../vocab/ab.syms -osymbols=../vocab/ab.syms ../fsts/miniL.fst ../fsts/miniL.binfst
#draw miniL.fst
fstdraw --isymbols=../vocab/ab.syms --osymbols=../vocab/ab.syms -portrait ../fsts/miniL.binfst | dot -Tpng >../images/miniL.png

#compile L.fst
fstcompile -isymbols=../vocab/chars.syms -osymbols=../vocab/chars.syms ../fsts/L.fst ../fsts/L.binfst

#compile miniV.fst
fstcompile -isymbols=../vocab/chars.syms -osymbols=../vocab/words.syms ../fsts/miniV.fst ../fsts/miniV.binfst
#draw miniV.fst
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms -portrait ../fsts/miniV.binfst | dot -Tpng >../images/miniV.png

#compile V.fst
fstcompile -isymbols=../vocab/chars.syms -osymbols=../vocab/words.syms ../fsts/V.fst ../fsts/V.binfst

#remove <epsilon>
fstrmepsilon ../fsts/L.binfst ../fsts/L.binfst
fstrmepsilon ../fsts/V.binfst ../fsts/V.binfst
fstrmepsilon ../fsts/miniV.binfst ../fsts/miniV.binfst
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms -portrait ../fsts/miniV.binfst | dot -Tpng >../images/miniV_rmeps.png

#determinize
fstdeterminize ../fsts/V.binfst ../fsts/V.binfst
fstdeterminize ../fsts/miniV.binfst ../fsts/miniV.binfst
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms -portrait ../fsts/miniV.binfst | dot -Tpng >../images/miniV_deter.png

#minimize
fstminimize ../fsts/V.binfst ../fsts/V.binfst
fstminimize ../fsts/miniV.binfst ../fsts/miniV.binfst
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms -portrait ../fsts/miniV.binfst | dot -Tpng >../images/miniV_min.png

#sort L.fst and V.fst
fstarcsort --sort_type=olabel ../fsts/L.binfst ../fsts/sortedL.binfst
fstarcsort --sort_type=ilabel ../fsts/V.binfst ../fsts/sortedV.binfst

fstarcsort --sort_type=olabel ../fsts/miniL.binfst ../fsts/miniL.binfst
fstarcsort --sort_type=ilabel ../fsts/miniV.binfst ../fsts/miniV.binfst

#compose sorted L.fst and V.fst to S.fst
fstcompose ../fsts/sortedL.binfst ../fsts/sortedV.binfst ../fsts/S.binfst
fstcompose ../fsts/sortedL.binfst ../fsts/sortedV.binfst ../fsts/LV.binfst
fstprint -isymbols=../vocab/chars.syms -osymbols=../vocab/words.syms ../fsts/S.binfst > ../fsts/S.fst

#compose miniL.fst and miniV.fst to miniS.fst
fstcompose ../fsts/miniL.binfst ../fsts/miniV.binfst ../fsts/miniS.binfst
#draw miniS.fst
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms -portrait ../fsts/miniS.binfst | dot -Tpng>../images/miniS.png

## ------------------------------------------------------------------------------------------------------------------------------------- ##

# compile M.fst and N.fst
fstcompile -isymbols=../vocab/chars.syms -osymbols=../vocab/chars.syms ../fsts/M.fst ../fsts/M.binfst
fstcompile -isymbols=../vocab/chars.syms -osymbols=../vocab/chars.syms ../fsts/N.fst ../fsts/N.binfst

# compose M,L,N 
fstcompose ../fsts/M.binfst ../fsts/L.binfst ../fsts/ML.binfst
fstcompose ../fsts/ML.binfst ../fsts/N.binfst ../fsts/MLN.binfst


fstshortestpath ../fsts/MLN.binfst | fstprint -isymbols=../vocab/chars.syms -osymbols=../vocab/chars.syms --show_weight_one

## ------------------------------------------------------------------------------------------------------------------------------------- ##

# compile E.fst and create EV.fst spell checker
fstcompile -isymbols=../vocab/chars.syms -osymbols=../vocab/chars.syms ../fsts/E.fst ../fsts/E.binfst
fstarcsort --sort_type=olabel ../fsts/E.binfst ../fsts/E.binfst
fstcompose ../fsts/E.binfst ../fsts/V.binfst ../fsts/EV.binfst


## ------------------------------------------------------------------------------------------------------------------------------------- ##

#compile W.fst and create LVW and EVW spell checkers
fstcompile -isymbols=../vocab/words.syms -osymbols=../vocab/words.syms ../fsts/W.fst ../fsts/W.binfst
fstarcsort --sort_type=ilabel ../fsts/W.binfst ../fsts/W.binfst

fstcompose ../fsts/sortedL.binfst ../fsts/sortedV.binfst ../fsts/LV.binfst
#sort,minimize etc. LV.fst
fstarcsort --sort_type=olabel ../fsts/LV.binfst ../fsts/LV.binfst
fstrmepsilon ../fsts/LV.binfst ../fsts/LV.binfst

fstcompose ../fsts/EV.binfst ../fsts/W.binfst ../fsts/EVW.binfst
fstcompose ../fsts/LV.binfst ../fsts/W.binfst ../fsts/LVW.binfst



fstcompose ../fsts/miniV.binfst ../fsts/W.binfst ../fsts/VW.binfst
fstdraw --isymbols=../vocab/chars.syms --osymbols=../vocab/words.syms -portrait ../fsts/VW.binfst | dot -Tpng >../images/VW.png

## ---------------------------------------------------------------------------------------------------------------------------------- ##
fstcompile -isymbols=../vocab/chars.syms -osymbols=../vocab/chars.syms ../fsts/E_laplace.fst ../fsts/E_laplace.binfst
fstarcsort --sort_type=olabel ../fsts/E_laplace.binfst ../fsts/E_laplace.binfst
fstcompose ../fsts/E_laplace.binfst ../fsts/V.binfst ../fsts/EV_laplace.binfst