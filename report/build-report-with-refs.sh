#!/usr/bin/env bash
set -euv

docker run --rm -v "$PWD":/data mrchoke/texlive bash -c "cd data && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex"
