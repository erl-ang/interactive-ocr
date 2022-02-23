#!/bin/bash
for f in ds_pdfs/*.pdf; do
	echo $f
	echo $(basename "$f" .pdf)
	gs -dBATCH -dNOPAUSE -sDEVICE=pnggray -r300 -dTextAlphaBits=4 -sOutputFile="tmp/$(basename "$f" .pdf)-%03d.png" $f  -c quit
	for p in tmp/*.png; do
		python ocrpy.py --file "$p"
		echo $(basename "$p")
		tesseract  x/$(basename "$p") text/$(basename "$p" .png) --psm 1 --oem 1
		awk 'BEGIN {RS=""}{gsub(/-\n/,"",$0); print $0}' text/$(basename "$p" .png).txt > text/tmp.txt && mv text/tmp.txt text/$(basename "$p" .png).txt
                cat text/$(basename "$p" .png).txt >> compiled/$(basename "$f" .pdf).txt
		rm $p
                #rm x/$(basename "$p")
	done
done