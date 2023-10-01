import csv
import os
from typing import List

labelcodec = dict()
with open('coco_labels_2017.csv', newline='') as codec:
    codecreader = csv.DictReader(codec, ('id', 'category', 'super category'))
    for row in codecreader:
        (id, idx) = row['id'].split(':')
        if row['category'] is not None:
            (cat, label) = row['category'].split(':')
        labelcodec[int(idx)] = label

with open('counts.csv', newline='') as counts:
    countsreader = csv.reader(counts, delimiter=',')
    headers = next(countsreader)
    row: list[str]
    with open("labelled-count-object-types.csv", "w") as data:
        data.write(f"category_label,distinct_count,total_count{os.linesep}")
        for row in countsreader:
            (category_id, distinct_count, total_count) = row
            if int(category_id) in labelcodec:
                label = labelcodec[int(category_id)]
                data.write(f"{str(label)}, {distinct_count}, {total_count}{os.linesep}")