# Proposal

## Motivation
In this project we will solve an object detection task. Object detection deals 
with detection of multiple objects in one picture. This task gains more 
importance as more smart solutions like autonomous cars, surveillance drones 
in cases of disasters or in mobile applications are being used every day.

There are many open problems in the area of object detection. Main open 
problems are listed and described below.

__Understand the relationships between detected objects [1]__ 
- It is easy for people to determine these relationships by just one look, but
it is not so easy for computers.

__Detection of background objects [2]__ 
- Smaller objects in the background are usually considered as a whole and not 
as different objects. This can be a problem in many applications.

__Class imbalance [3]__ 
- Some objects in the dataset used for training might be more frequent than 
others which might result in a model which is less accurate at classifying 
less frequent objects.

__Spatial imbalance [3]__
- Bounding boxes might not be accurately found which might influence the 
accuracy of the classification.

__Objective imbalance [3]__ 
- Object detection performs regression to find bounding boxes and 
classification 
to classify objects inside them and one of these tasks might train faster than 
the other.

__Incrementally learn, detecting of new classes [4]__ 
- Humans are good at creating new objects in existing object classes, such as
new types of clothes. It might be problematic to incrementally learn to 
classify these new objects due to the lack of new available data without 
losing accuracy on already trained objects.

## Related Work
There are many solutions for object detection task. Main approaches are based
on Single Shot MultiBox Detector (SSD), Region-based Convolutional Neural 
Networks (R-CNN) and You Look Only Once (YOLO). Some systems based on R-CNN 
and YOLO are described below. 

In 2016 an approach based on Region-based Fully Convolutional Networks (R-FCN)
for object detection with 101-layer ResNet as a backbone was used in [5]. 
Authors were able to achieve similar results as Faster R-CNN with 101-layer
ResNet on COCO dataset, but R-FCN approach is much faster during both training
and inference. The best achieved mAP (mean Average Precision)
is 53.2 % / 31.5 % (AP@[0.5] / AP@[0.5, 0.95]).

In 2016 an approach known as YOLO (You Only Look Once) was used in [6]. This
approach is able to predict all bounding boxes in the entire image in one 
evaluation. It is based on splitting input image into S x S grid and then 
performing prediction in every grid, however using entire image for that 
prediction. It is a very fast approach, capable of processing 155 images per 
second when faster version of algorithm is used and 45 images per second when 
slower but more accurate version is used. When comparing to other real-time 
systems, the best achieved mAP is 63.4 % which was state-of-the-art result 
achieved on PASCAL-VOC. However, at less-than-real-time YOLO with VGG-16 as a 
backbone was outperformed by Faster R-CNN also with VGG-16 as a backbone.

In 2016 authors [7] said that, there are two important components in object 
detection: a feature extractor and object classifier. In [7] authors focused
on showing importance of classification of objects. They used Networks on 
Convolutional feature maps (NoCs), approach that uses shared, 
region-independent convolutional features. With this approach and 101-layer 
ResNet as a backbone, they were able to win ImageNet and COCO challenges 2015. 
The best achieved AP@0.5 is 48.4 % on COCO dataset.

In 2016 Multi-Scale Convolutional Neural Network (MS-CNN) was used in [8] 
for real-time object detection. MS-CNN consists of two sub-networks, proposal 
sub-network and detection sub-network. They achieved state-ot-the-art 
results on datasets KITTI and Caltech for up to 15 images per second.

In 2016 Region Proposal Network (RPN) for predicting object bounds and 
objectness score at each position was used in [9]. RPN is used to generate 
region proposals, which are used as input to Fast R-CNN. When using VGG-16 
model, their system achieved state-of-the-art results on PASCAL VOC 2007 
(73.2 % mAP) and 2012 (70.4 % mAP).

In 2017 an approach based on pyramidal hierarchy of deep convolutional 
networks to build feature pyramids was used in [10]. They used top-down 
architecture with lateral connections, called Feature Pyramids Networks (FPN) 
to build semantic feature maps at all scales. Then they used FPN in Faster 
R-CNN system and achieved state-of-the-art single-model results on COCO dataset.
Backbone was 101-layer ResNet. The best achieved mAP is 
59.1 % / 36.2 % (AP@[0.5] / AP@[0.5, 0.95]).

## Datasets
We identified multiple available datasets designed for object detection task. 
Best known and most used are listed and described below.

__ImageNet [11]__ 
- Data set is not publicly available, access has to be provided. 
- Number of images with bounding box annotations is 1,034,908.

__Open Images Dataset [12]__ 
- Approximately 8 boxed objects per image.
- Own detection metric.

__MS-COCO [13]__ 
- 5 captions per image.
- Annotations for test dataset are not publicly available.
- Own detection metric.

| Open Images Dataset |    Train   | Validation |   Test  | # Classes |
|:-------------------:|:----------:|:----------:|:-------:|:---------:|
|        Images       |  1,743,042 |   41,620   | 125,436 |     -     |
|        Boxes        | 14,610,229 |   303,980  | 937,327 |    600    |

| MS-COCO |  Train  | Validation |  Test  | # Classes |
|:-------:|:-------:|:----------:|:------:|:---------:|
|  Images | 118,000 |    5,000   | 41,000 |     -     |
|  Boxes  |    -    |      -     |    -   |     80    |

Each of the above datasets consists of images annotated with multiple 
bounding boxes labeled with a class name. Most of the data was annotated 
manually by humans and labels were selected as specifically as possible. 
It should be noted that the label distribution can be heavily skewed.

## High-Level Solution Proposal
Based on the analysis of the problem area we propose the following method. 

To detect objects we will use approach known as YOLO (You Only Look Once). 
YOLO approach is based on creating a grid in an image, detecting objects in 
grid cells and classifying the detected objects. All this happens by just one 
look at the image which is very fast. We chose this approach due to its 
simplicity, efficiency and speed.

Our architecture of the model will be based on CNNs (Convolutional Neural 
Networks). We chose this type of neural networks due to their excellent 
properties in computer vision problems. If we have enough time we would like 
to try more advanced architectures using skip connections  for the possibility 
to train deeper models or using dilated convolutions to get bigger receptive 
field with less parameters.

For training, validation and testing we will use COCO and Open Images Dataset 
(OID). Since, COCO is smaller than OID we will use it in the beginning of 
development. OID will be used in the latter stage. We would like to use whole 
OID, but  due to technical limitations we can not use the entire dataset, 
but only its small part containing small subset of objects.

## References
[1] https://www.quora.com/What-are-the-research-problems-in-object-detection

[2] https://www.frontiersin.org/articles/10.3389/frobt.2015.00029/full#h5

[3] https://arxiv.org/pdf/1909.00169.pdf?fbclid=IwAR1NnXoy3iB0-4NHISp-DHq8rciPLpPPUlRMUyCYqVAzxatXXGTWFHn57sQ

[4] https://www.frontiersin.org/articles/10.3389/frobt.2015.00029/full#h5

[5] https://papers.nips.cc/paper/6465-r-fcn-object-detection-via-region-based-fully-convolutional-networks.pdf

[6] https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf

[7] https://ieeexplore.ieee.org/abstract/document/7546875

[8] https://link.springer.com/chapter/10.1007/978-3-319-46493-0_22

[9] https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf

[10] http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf

[11] http://www.image-net.org/

[12] https://storage.googleapis.com/openimages/web/index.html

[13] http://cocodataset.org/#home
