This repository contains the code for the **SenSys 2023** submitted paper: *"[BiGuide: A Bi-level Data Acquisition Guidance for Object Detection on Mobile Devices]"*. 

The BiGuide implementation is based on [mmyolo](https://github.com/open-mmlab/mmyolo).

## Outline
* [I. Prerequisites](#1)
* [II. Post-experiment Survey Questions](#2)
* [III. User Study Instructions](#3)
* [IV. Dataset](#4)
* [V. Running BiGuide on Client and Server Devices](#5)


#### <span id="1">I. Prerequisites
Setup the [mmyolo](https://github.com/open-mmlab/mmyolo) prerequisites.

#### <span id="2">II. Post-experiment Survey Questions
  * Q1: The BiGuide guidance content was easy to understand.
  * Q2: The BiGuide guidance itself was easy to follow.
  * Q3: The BiGuide guidance displayed fast after pressing the "take the image button".
  * Q4: BiGuide was helpful for collecting diverse data.
  * Q5: FreGuide was helpful for collecting diverse data.
  * Q6: BiGuide made me confident in that I was collecting useful data.
  * Q7: FreGuide made me confident in that I was collecting useful data.
  * Q8: BiGuide made me change poses, locations, and angles more frequently.
    
#### <span id="3">III. User Study Instructions
Follow the instructions in indoor scenario bellow:
##### Intro:
The goal of this experiment is to collect diverse data to train a machine learning model that can detect the class and the bounding box of indoor objects. Seven indoor objects are included in this experiment. They are located in 7 different places in the lab. 
##### Instructions:
Users will collect images by two systems: FreGuide and BiGuide.
For FreGuide:
* Users hold the phones in portrait  mode, taking pictures in front of the object and moving the phone. 
* Users can tilt their phone or change their position as they want.
* Users need to take 20 images per object. 
For BiGuide system:
* Users hold the phones in portrait  mode, taking pictures in front of the object and moving the phone. 
* Users can tilt their phone or change their position as they want.
* When receiving guidance from the system, users need to follow the guidance to take the photo.
* Users need to take 20 images per object. 
Note: After the user study, the model will be trained offline using collected data to compare their performance.

#### <span id="4">IV. Dataset
Our collected dataset and the annotations are in "./Server/data/"

#### <span id="5">V. Running BiGuide on Client and Server Devices
##### 1. Running BiGuide on Server
```
 uvicorn UserStudy_BiGuide_indoor:app --reload --host xxx.xxx.x.x --port xxxx
 ```
 ##### 2. Running BiGuide on Mobile Devices
Directly build the app through Unity with the code in "APP.zip"
