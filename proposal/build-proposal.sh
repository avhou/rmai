#!/usr/bin/env bash
set -euv

docker run --rm -v "$PWD":/data mrchoke/texlive bash -c "cd data && pdflatex proposal.tex"
