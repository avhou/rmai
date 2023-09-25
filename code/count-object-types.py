import json
import sys
from typing import *
import os

class Count:
    object_type: int
    total_count: int
    images: set[int]

    def __init__(self, object_type: int):
        self.object_type = object_type
        self.images = set()
        self.total_count = 0

    def add_file(self, image_id: int):
        self.total_count = self.total_count + 1
        self.images.add(image_id)

    def distinct_count(self) -> int:
        return len(self.images)

    def __str__(self):
        return f'{self.object_type},{self.distinct_count()},{self.total_count}'

    def __repr__(self):
        return f'Count object_type {self.object_type}, distinct count {self.distinct_count()}, total count {self.total_count}'

def str_to_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None

def generate_counts(object_types_str: List[str]):
    object_types = [item for item in set(map(str_to_int, object_types_str)) if item is not None]
    counts = dict()
    for object_type in object_types:
       counts[object_type] = Count(object_type)

    print(f"looking for object types {object_types}")
    print("reading input file")
    with open("data/annotations/instances_train2017.json") as data:
        print("parsing input file to json")
        json_data = json.load(data)
        print("parsing done")
        for annotation in json_data["annotations"]:
            category = int(annotation["category_id"])
            image_id = int(annotation["image_id"])
            count = counts.get(category)
            if count is not None:
                count.add_file(image_id)
                counts[category] = count
    print("writing output")
    with open("counts.csv", "w") as data:
        data.write(f"category_id,distinct_count,total_count{os.linesep}")
        for count in sorted(counts.keys()):
            data.write(f"{str(counts[count])}{os.linesep}")



if __name__ == '__main__':
    generate_counts(sys.argv[1:])
    print("done")
