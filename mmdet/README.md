# Cube World Scanning: Character Detection in Minecraft with FCOS and YOLO

## Task description

This project explores the possibilities of modern object detection models using the example of the Minecraft game world. We will have to retrain the detectors on a specific game dataset and compare them in terms of accuracy, speed and quality of predictions.

## Goal

Two models need to be retrained: EKAS and YOLO to recognize the main classes of mobs in Minecraft (for example, cow, zombie, etc.).
It is necessary to conduct an inference of the models on the video stream in order to visually evaluate the work of the models, as well as compare them by key metrics: precision, recall, mAP и FPS. You also need to implement additional metrics and measure them for the models.

## Dataset

The dataset is in the PASCAL VOC XML format..

This dataset stores classes describing the main characters in Minecraft.:

```
CLASSES (17)
'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
```
