## Specifications in study guide

You can make use of databases of already annotated images.
You can annotate your own set of images.
The aim of this topic will be to find out how to best construct the training set. In your simulations, train the model with training sets that differ in one dimension, keeping everything else constant.

Research questions that you could address are : 

- What is the minimum number of images required for good training? Does this depend on the type of object that you are trying to track?
- What is the influence of the variability of the object? When the goal is to detect one specific bowl in various contexts, will training be better than if you have to be able to detect different shapes of bowls?
- What is the influence of the variability of the background of the object? If you always detect the same target object, does it matter in what contexts it appears?
- What is the influence of the size of the training images? And the spatial resolution of the images? Or the contrast of the images?
- Does it matter whether the target objects more commonly appears in the middle of the image or away from the middle? What is the influence of the variation in the distance (i.e., size in the image) towards the object in the image?
- Does it matter how tight the bounding box is? Should one spend the extra time to get the bounding boxes just right, or is it better to annotate more images, but care less about the bounding box?
- If the target object is partially occluded, should one annotate the entire region of where the object is in the image (including the occluded part) or just the visible part? How does this work for objects hidden behind a transparent object? Should these still be annotated for training?

## Sources on YOLO

[Datacamp](https://www.datacamp.com/blog/yolo-object-detection-explained?utm_source=google&utm_medium=paid_search&utm_campaignid=20454548123&utm_adgroupid=158489640211&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=674693108598&utm_targetid=aud-299261629574:dsa-2191608054892&utm_loc_interest_ms=&utm_loc_physical_ms=1001091&utm_content=DSA~blog~Data-Science&utm_campaign=230119_1-sea~dsa~tofu-blog_2-b2c_3-eu_4-prc_5-na_6-na_7-le_8-pdsh-go_9-na_10-na_11-na-sep23&gclid=Cj0KCQjw9rSoBhCiARIsAFOiplkOsVd-6CVVpOUUElzzrRzdG-5GrxFNK5CfQu8eI_iSexidZ4sDZwIaAnQfEALw_wcB)

[Youtube tutorial](https://www.youtube.com/watch?v=_FNfRtXEbr4)

## On training

### Google Collab

[Google collab](https://colab.research.google.com/?utm_source=scs-index)

Colab, or "Colaboratory", allows you to write and execute Python in your browser, with

- Zero configuration required
- Access to GPUs free of charge
- Easy sharing
