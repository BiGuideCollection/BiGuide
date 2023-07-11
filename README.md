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
Or directly use mmyolo_env.yml to set up the environment.
```
conda env create -f mmyolo_env.yml
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
During the user study, each participant began by reviewing a set of instructions we have prepared. The instructions for indoor scenario are the same as the instruction for wildlife exhibits scenario. We take the instructions in indoor scenario below for instance.
##### Intro
The goal of this experiment is to collect diverse data to train a machine-learning model that can detect the class and the bounding box of indoor objects. Seven indoor objects are included in this experiment. They are located in 7 different places in the lab. 
##### Instructions
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
We share the data collected using a commodity Google Pixel 3 XL smartphone during the user study in the indoor scenario and wildlife exhibits scenario.
##### Indoor scenario: 
We set up the indoor scenario in a typical office environment. 10 users were guided to collect 20 images for each object in this environment. We included seven object classes: mobile phone, scissors, ball, tin can, light bulb, mug, and remote control. These objects were placed in seven distinct locations within a controlled environment with stable lighting conditions (see Figure 1). Users moved around different locations to collect the images of the objects. 
<p align="center">
  <img src="https://github.com/BiGuideCollection/BiGuide/assets/138166113/7abc914f-ea9e-4a2a-9c44-9c7d1d63c6b9" width="500" title="Figure 1. Example images of 7 objects positioned in 7 locations in the indoor scenario. These objects were placed in a controlled environment with stable lighting conditions."/>
</p>

##### Wildlife exhibits scenario: 
We set up the wildlife exhibits scenario in a local wildlife center. This scenario involved outdoor scenes with dynamic objects, specifically lemurs. 10 users were tasked with capturing 20 images for each lemur species. Three lemur species were showcased in the center: blue-eyed black lemur, ring-tailed lemur, and red ruffed lemur, as depicted in Figure 2. Different lemur species were housed in separate exhibits within the center, requiring users to move between the exhibits. Users’ visits were scheduled at different times on seven different days, aligning with the center’s general tour schedule. This led to users encountering different weather conditions, including sunny and heavily rainy days. On sunny days, the lemurs were more active, engaging in activities like climbing and exploring; on rainy days, the lemurs tended to gather and rest inside their cages. Compared to the images collected in the indoor scenario, the wildlife images present greater complexity and detection challenges due to the lemurs’ varied poses and sizes, occlusion from cages, and unstable lighting conditions.
<p align="center">
  <img src="https://github.com/BiGuideCollection/BiGuide/assets/138166113/6d1495ec-116f-48cc-945d-adf97526ea6d" width="500" title="Figure 2.Example images of 3 lemur species enclosed in 3 distinct exhibits in the wildlife exhibits scenario. Images obtained in this scenario are more complex due to lemurs’ varying poses and sizes, as well as the diverse backgrounds."/>
</p>

Each user collected 20 images for each object in their assigned scenario. We manually labeled all data collected by users (4400 images in total). For the test set, we pre-collected 110 images for each class to capture images under varying lighting and weather conditions to ensure fairness in the evaluation results. In total, we amassed 770 images in the indoor test set and 330 images in the wildlife test set. Our collected data samples and the annotations samples are in "./data/". Will release the full dataset in Oct.2023.
The indoor dataset has the same structure as the wildlife exhibits dataset. We take the structure of indoor dataset below for instance:
```
indoor_coco/
  -annotations/
    -test_precollected_indoor.json
    -train_biguide_indoor_user1.json
    -train_biguide_indoor_user2.json
    -...
  -images/
    -target_2023_MM_DD_hh_mm_ss_XXXXXX.jpg
    -...
  -class_with_id.txt
```
The images are named by the time when they were captured.

#### <span id="5">V. Running BiGuide on Client and Server Devices
We take the indoor scenario as an example.
##### 1. Running BiGuide on Server
Change the path in UserStudy_BiGuide_indoor.py to your own path. Then run the command below.
```
cd tools/
uvicorn UserStudy_BiGuide_indoor:app --reload --host xxx.xxx.x.x --port xxxx
```
Replace ```--host xxx.xxx.x.x --port xxxx``` with your own IP address and port number.

##### 2. Running BiGuide on Mobile Devices
When the code on the server is running, you can build the mobile app through Unity with the code in "APP.zip"
