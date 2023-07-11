This repository contains the code for the **SenSys 2023** submitted paper: *"BiGuide: A Bi-level Data Acquisition Guidance for Object Detection on Mobile Devices"*. 

The BiGuide implementation is based on [mmyolo](https://github.com/open-mmlab/mmyolo).

## Outline
* [I. Prerequisites](#1)
* [II. Post-experiment Survey Questions](#2)
* [III. User Study Instructions](#3)
* [IV. Dataset](#4)
* [V. Running BiGuide on Client and Server Devices](#5)


#### <span id="1">I. Prerequisites
Setup the [mmyolo](https://github.com/open-mmlab/mmyolo) prerequisites as follows.
```
conda create -n mmyolo python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmyolo
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc6,<3.1.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```

#### <span id="2">II. Post-experiment Survey Questions
We assembled a set of questions in different categories for the post-experiment survey to gather feedback. For the category of data acquisition guidance (Q1-Q3), we asked the participants if the designed guidance was easy to understand and follow and the guidance generation was fast. For the category of system experience (Q4-Q8), we asked the participants if the system was helpful and if they felt more confident and more involved when using the system. All questions in these categories were answered on a five-point Likert scale. At the end of the survey, we asked the participants to identify their favorite system and to leave open-ended feedback about the overall experience.
  * Q1: The BiGuide guidance content was easy to understand.
  * Q2: The BiGuide guidance itself was easy to follow.
  * Q3: The BiGuide guidance displayed fast after pressing the "take the image button".
  * Q4: BiGuide was helpful for collecting diverse data.
  * Q5: FreGuide was helpful for collecting diverse data.
  * Q6: BiGuide made me confident in that I was collecting useful data.
  * Q7: FreGuide made me confident in that I was collecting useful data.
  * Q8: BiGuide made me change poses, locations, and angles more frequently.
    
#### <span id="3">III. User Study Instructions
During the user study, each participant began by reviewing a set of instructions we have prepared. The instructions for indoor scenario are the same as the instruction for wildlife exhibits scenario. We take the instructions in indoor scenario below for instance:
##### Intro:
The goal of this experiment is to collect diverse data to train a machine-learning model that can detect the class and the bounding box of indoor objects. Seven indoor objects are included in this experiment. They are located in 7 different places in the lab. 
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
We share the data collected during the user study. Our collected data samples and the annotations samples are in "./Server/data/". Will release the full dataset in Oct.2023.
The indoor dataset has the same structure as the wildlife exhibits dataset. We take the structure of indoor dataset below for instance:
```
indoor_coco/
  -annotations/
    -test_precollected_indoor.json
    -train_biguide_indoor_user1.json
    -train_biguide_indoor_user2.json
    -...
  -images/
    -image1.jpg
    -image2.jpg
    -...

```

#### <span id="5">V. Running BiGuide on Client and Server Devices
##### 1. Running BiGuide on Server
```
 uvicorn UserStudy_BiGuide_indoor:app --reload --host xxx.xxx.x.x --port xxxx
 ```
 ##### 2. Running BiGuide on Mobile Devices
Directly build the app through Unity with the code in "APP.zip"
