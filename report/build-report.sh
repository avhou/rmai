rm report.pdf;docker run --rm -v "$PWD":/data mrchoke/texlive bash -c "cd data && pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex";open report.pdf
