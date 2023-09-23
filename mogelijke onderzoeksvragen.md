## Research question ideas

### Detecting single person vs persons in a crowd

First version of YOLO had difficulties with this task. The underlying reason was that each grid in the YOLO architecture is designed for single object detection.

### Does vantage point affects object detection

Broadly three categories could be used:

- Above horizon
- Near horizon
- Below horizon

### Does variation in object shapes influence object detection

Open versus closed books

### Does variation in image coloring influence object detection

Same kind of objects but colored differently.

### Can seasons be detected from image background

### How does bias in training set influence outcome

### How does fog affect object detection

[YOLO-GW: Quickly and Accurately Detecting Pedestrians in a Foggy Traffic Environment](https://mdpi-res.com/sensors/sensors-23-05539/article_deploy/sensors-23-05539.pdf?version=1686649274)

This paper reports on an adaptation of YOLOv7 to improve detection of pedestrians in foggy traffic conditions. The treatment seems too complex for the setting of this course. But maybe another related research question can be formulated. Given following two categories:

- Persons wearing a fluorescent vest
- Persons not wearing a fluorescent vest

How does a fluorescent vest affect object detection in YOLOv7 (without adaptations)?   

related : how does wearing a facemask affect object detection?
