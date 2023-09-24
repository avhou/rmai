## Research question ideas

### Can YOLO be trained such that it detects the difference between sea turtles and land turtles

The challenge here could be that sea turtles and land turtles have a high degree of resemblance. Moreover I think that there are lots of pictures available online.

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

### != object detection?

YOLOv8 supports multiple vision tasks such as object detection, segmentation, pose estimation, tracking, and classification

### textbook suggestions

Research questions that you could address are :

- What is the minimum number of images required for good training? Does this depend on the type of object that you are trying to track?
- What is the influence of the variability of the object? When the goal is to detect one specific bowl in various contexts, will training be better than if you have to be able to detect different shapes of bowls?
- What is the influence of the variability of the background of the object? If you always detect the same target object, does it matter in what contexts it appears?
- What is the influence of the size of the training images? And the spatial resolution of the images? Or the contrast of the images?
- Does it matter whether the target objects more commonly appears in the middle of the image or away from the middle? What is the influence of the variation in the distance (i.e., size in the image) towards the object in the image?
- Does it matter how tight the bounding box is? Should one spend the extra time to get the bounding boxes just right, or is it better to annotate more images, but care less about the bounding box?
- If the target object is partially occluded, should one annotate the entire region of where the object is in the image (including the occluded part) or just the visible part? How does this work for objects hidden behind a transparent object? Should these still be annotated for training?

### discussion 20230924

- variations in size of dataset and/or object type (filtered dataset)
- variations in hyperparameters of training
- assemble a custom dataset, select bounding box + label the images ourselves

coco object types : 
12/13 street signs/stop signs
19/24 horse / zebra
41/42 skateboard / surf board
48/49/50 fork knife spoon
78/79 microwave/oven

research question :
we investigate the classification of closely resembling object types.  
variations in : 
- training set size
- hyperparameters (literature)
- contrast / illumination / resolution
- yolo versions?
