#!/usr/bin/env bash
set -euv

if [ ! -f data/annotations/instances_train2017.json ]; then
    echo "File not found, creating folder"
    mkdir data
    echo "Downloading file"
    curl "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" > data/annotations_trainval2017.zip
    unzip -d data data/annotations_trainval2017.zip
fi

python3 count-object-types.py 12 13 19 24 41 42 48 49 50 78 79

# duckdb
# cat counts.sql | duckdb
