# BiGuide: A Bi-level Data Acquisition Guidance for Object Detection on Mobile Devices

This repository contains download links and the introduction of our collected indoor and lemur datasets, as well as the code for IPSN 2024 paper: ["BiGuide: A Bi-level Data Acquisition Guidance for Object Detection on Mobile Devices"]() by [Lin Duan](https://scholar.google.com/citations?user=3KGmyogAAAAJ&hl=en), [Ying Chen](https://scholar.google.com/citations?hl=en&user=aoMpAKoAAAAJ), Zhehan Qu, Megan McGrath, Erin Ehmke, and [Maria Gorlatova](https://maria.gorlatova.com/bio/). 

## Outline:
* [Overview](#1)
* [Demo Video](#2)
* [Dataset](#3)
* [BiGuide Implementation](#4)
* [Related Materials](#5)
* [Citation](#6)
* [Acknowledgments](#7)

The rest of the repository is organized as follows. [**Section 1**](#1) gives a brief overview of BiGuide. [**Section 2**](#2) shows a short demo video of the data collection process. [**Section 3**](#3) introduces our own collected dataset. [**Section 4**](#4) briefly introduces the implementation of the BiGuide. [**Section 5**](#5) shows the post-experiment survey questions and user study instructions. The citation information, author contacts, and acknowledgments are introduced in [**Section 6**](#6) and [**Section 7**](#7). 

## 1. <span id="1"> Overview</span> 
<p align="center"><img src="https://github.com/BiGuideCollection/BiGuide/blob/main/images/system_design.png" width="580"\></p>
<p align="center"><strong>Figure 1. Overview of BiGuide design.</strong></p> 

BiGuide is a data acquisition system that instructs users in collecting diverse and informative data for training OD models. An overview of BiGuide is shown in Figure 1. BiGuide comprises two major components: data importance estimation and guidance generation and adaptation. They are deployed on the edge server such that no significant computation overhead is introduced on the mobile device. In addition to the server, there is a mobile app running on a mobile device which wirelessly sends images captured by the user to the edge, receives real-time data acquisition guidance from the edge, and presents it to the user.

## 2. <span id="2"> Demo Video</span>

A short demo video of collecting images using BiGuide is shown below. The demo is performed using a Google Pixel 3 XL mobile phone running Android 11. A Lenovo laptop with an AMD Ryzen 4700H CPU and an NVIDIA GTX 1660 Ti GPU serves as the edge server. The mobile app running on the mobile device sends images to the edge in real time. The edge generates the data acquisition guidance and sends it wirelessly to the mobile device.

**Note: You can find the full demo video (2023 version) on our website by clicking the gif image below.**

[![Demo](https://github.com/BiGuideCollection/BiGuide/blob/main/images/IMG_3860.gif)](https://sites.duke.edu/linduan/)

## 3. <span id="3"> Dataset</span>
We share the data collected using a commodity Google Pixel 3 XL smartphone during the user study in the **indoor scenario** and **wildlife exhibits scenario**. The detailed information about the collected datasets is presented below.

### 3.1. Indoor Scenario
We set up the indoor scenario in a typical office environment. Users were guided to collect **20 images for each object** in this environment. We included **seven object classes**: mobile phone, scissors, ball, tin can, light bulb, mug, and remote control. These objects were placed in **seven distinct locations** within a controlled environment. Users moved around different locations to collect the images of the objects. 24 images have been removed to address privacy concerns. The details of the training set are summarized in the table below:

<table border="0">
    <tr>
        <td>Number of BiGuide users</td><td>10</td> <td>Number of FreGuide users</td><td>10</td> <td>Number of CovGuide users</td><td>1</td>
    </tr>
    <tr>
        <td>Number of object classes</td><td>7</td> <td>Number of object classes</td><td>7</td> <td>Number of object classes</td><td>7</td>
    </tr>
    <tr>
        <td>Number of images per class</td><td>20</td> <td>Number of images per class</td><td>20</td> <td>Number of images per class</td><td>20</td>
    </tr>
    <tr>
        <td><b>Total images</b></td><td>10 x 7 x 20 - 15 = 1,387</td> <td><b>Total images</b></td><td>10 x 7 x 20 - 11 = 1,389</td> <td><b>Total images</b></td><td>1 x 7 x 20 = 140</td>
    </tr>
</table>

To evaluate the performance of models trained on the collected data, we pre-collected 110 images for each class under varying conditions to ensure fairness in the evaluation results. In total, we amassed 770 images in the indoor test set. 2 images have been removed to address privacy concerns.

#### Examples of images in the indoor dataset:
<p align="center">
  <img src="https://github.com/BiGuideCollection/BiGuide/assets/138166113/7abc914f-ea9e-4a2a-9c44-9c7d1d63c6b9" width="500" alt="Alt text" title="Figure 1. Example images of 7 objects positioned in 7 locations in the indoor scenario. These objects were placed in a controlled environment with stable lighting conditions.">
</p>
<p align="center">
    <em>Figure 1. Example images of 7 objects positioned in 7 locations in the indoor scenario. These objects were placed in a controlled environment with stable lighting conditions.</em>
</p>

### 3.2. Wildlife Exhibit Scenario
We set up the wildlife exhibit scenario in the Duke Lemur Center. This scenario involves outdoor scenes with dynamic objects, specifically lemurs. The Duke Lemur Center has various lemur enclosure styles, including forest habitats, but our project focused on a section of summer enclosures on the public tour path. **Users** were tasked with capturing **20 images for each lemur species**. **3 lemur species** were showcased in the center: blue-eyed black lemur, ring-tailed lemur, and red ruffed lemur. Different lemur species were housed in distinct enclosures, requiring users to move between these separate areas. Users’ visits were scheduled at different times on seven different days, aligning with the center’s general tour schedule. This led to users encountering different weather conditions, including sunny and heavily rainy days. On warm, sunny days, the lemurs were more active, engaging in activities like climbing and exploring; on cold, rainy days, the lemurs tended to gather and rest inside their cages. Compared to the images collected in the indoor scenario, the wildlife images present greater complexity and detection challenges due to the lemurs’ varied poses and sizes, occlusion from cages, and unstable lighting conditions. 23 images have been removed to address privacy concerns. The details of the training set are summarized in the table below:

<table border="0">
    <tr>
        <td>Number of BiGuide users</td><td>10</td> <td>Number of FreGuide users</td><td>10</td> <td>Number of CovGuide users</td><td>1</td>
    </tr>
    <tr>
        <td>Number of object classes</td><td>3</td> <td>Number of object classes</td><td>3</td> <td>Number of object classes</td><td>3</td>
    </tr>
    <tr>
        <td>Number of images per class</td><td>20</td>  <td>Number of images per class</td><td>20</td>  <td>Number of images per class</td><td>20</td>
    </tr>
    <tr>
        <td><b>Total images</b></td><td>10 x 3 x 20 - 15 = 585</td>  <td><b>Total images</b></td><td>10 x 3 x 20 - 6 = 594</td>  <td><b>Total images</b></td><td>1 x 3 x 20 - 2 = 58</td>
    </tr>
</table>

For the wildlife test set, we amassed 330 images in total to evaluate the performance of models trained on the collected data. 13 images have been removed to address privacy concerns.

#### Examples of images in the wildlife exhibit dataset:
<p align="center">
  <img src="https://github.com/BiGuideCollection/BiGuide/assets/138166113/6d1495ec-116f-48cc-945d-adf97526ea6d" width="500" alt="Alt text" title="Figure 2.Example images of 3 lemur species enclosed in 3 distinct exhibits in the wildlife exhibits scenario. Images obtained in this scenario are more complex due to lemurs’ varying poses and sizes, as well as the diverse backgrounds.">
</p>
<p align="center">
    <em>Figure 2.Example images of 3 lemur species enclosed in 3 distinct exhibits in the wildlife exhibits scenario. Images obtained in this scenario are more complex due to lemurs’ varying poses and sizes, as well as the diverse backgrounds.</em>
</p>

### 3.3. Download Indoor and Wildlife Exhibit Datasets
+ The indoor dataset can be downloaded via https://duke.box.com/s/r3wjlv4jtp83t4dnc1j9u1jp8nyqltgv
+ The wildlife exhibit dataset can be downloaded via https://duke.box.com/s/elcgsxonsyd39vou1swak7spm5mnh9w4

#### Hierarchical structure of the datasets:
We manually labeled all data collected by users. Our collected data samples and the annotations samples are in "./data/". Will release the full dataset in Mar.2024.
The indoor dataset has the same structure as the wildlife exhibit dataset. We take the structure of indoor dataset for instance.
The dataset follows a hierarchical file structure shown below. The two sub-folders, ***annotations*** and ***images***, correspond to the annotations and images of the indoor dataset.

- The tree structure of the dataset folder:
```
indoor_coco/
  -annotations/
    -train_biguide_indoor_user1.json
    -train_biguide_indoor_user2.json
    -...
  -images/
    -train_biguide_user1_0.jpg
    -train_biguide_user1_1.jpg
    -...
  -class_with_id.txt
  -indoor_coco_readme.txt
```
The images are named in the order of their capture sequence.

### 3.4. Data Distribution Comparision:
We compare the distribution of the data collected by CovGuide, FreGuide and BiGuide, and show the distribution comparisons in Figure 3 and Figure 4. We obtain the data distribution by (1) extracting the image features of all data; (2) using PCA to reduce them into 2 dimensions; and (3) drawing the data points. From the data distribution, we can observe that data collected by BiGuide are much more diverse than the data collected by other baseline systems.
<p align="center">
  <img src="https://github.com/BiGuideCollection/BiGuide/blob/main/images/indoor_fre(0.1029)_cov(0.0841).png" width="200" alt="Alt text" title="Figure 3. Data distribution of FreGuide and CovGuide for the indoor scenario.">
  <img src="https://github.com/BiGuideCollection/BiGuide/blob/main/images/indoor_fre(0.0923)_bi(0.0946).png" width="200" alt="Alt text" title="Figure 4. Data distribution of FreGuide and BiGuide for the indoor scenario.">
</p>
<p align="center">
    <em>Figure 3. Indoor scenario data distribution. Left: FreGuide v.s. CovGuide (small scale). Right: FreGuide v.s. BiGuide (large scale).</em>
</p>
<p align="center">
  <img src="https://github.com/BiGuideCollection/BiGuide/blob/main/images/lemur_fre(0.1750)_cov(0.1436).png" width="200" alt="Alt text" title="Figure 5. Data distribution of FreGuide and CovGuide for the wildlife exhibits scenario.">
 <img src="https://github.com/BiGuideCollection/BiGuide/blob/main/images/lemur_fre(0.0711)_bi(0.1447).png" width="200" alt="Alt text" title="Figure 6. Data distribution of FreGuide and BiGuide for the wildlife exhibits scenario.">
</p>
<p align="center">
    <em>Figure 4. Wildlife exhibits scenario data distribution. Left: FreGuide v.s. CovGuide (small scale). Right: FreGuide v.s. BiGuide (large scale).</em>
</p>

## 4. <span id="4"> BiGuide Implementation</span>

We implement BiGuide in an edge-based architecture using commodity smartphone as the mobile client. We design a mobile app on smartphones running Android 11 using Unity 2020.3.14f and ARCore 4.1.7. Data importance estimation and guidance generation and adaptation are executed on the edge server with an Intel i7 CPU, an NVIDIA 3080 Ti GPU, and 64GB DDR5-4800 RAM. For data importance estimation, we employ YOLOv5 for fast OD model inference. Communication between the server and smartphones occurs over one-hop 5~GHz WiFi (802.11n), with images resized to $3\times1480\times720$ and JPEG compressed to reduce latency.

### 4.1 Running BiGuide on Server
The BiGuide implementation is based on [mmyolo](https://github.com/open-mmlab/mmyolo). Please setup the [mmyolo](https://github.com/open-mmlab/mmyolo) prerequisites as follows.
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

Once the environment is set up, you can run BiGuide, using the indoor scenario as an example. Please modify the path in UserStudy_BiGuide_indoor.py to match your specific file location. After making this adjustment, execute the following command:
```
cd tools/
uvicorn UserStudy_BiGuide_indoor:app --reload --host xxx.xxx.x.x --port xxxx
```
Replace ```--host xxx.xxx.x.x --port xxxx``` with your own IP address and port number.

### 4.2. Running BiGuide on Mobile Devices
When the code on the server is running, you can build the mobile app through Unity with the code in "APP.zip". Please replace ```http://xxx.xxx.x.x:xxxx/guidance``` and ```http://xxx.xxx.x.x:xxxx/realtimeguidance``` with your own IP address and port number in line 42 and 43 of APP/UserStudyGuidance_indoor/Assets/Scripts/Screenshot.cs. Then, enjoy your data collection!

## 5. <span id="5"> Related Materials</span>

### 5.1 Post-experiment Survey Questions
We assembled a set of questions in different categories for the post-experiment survey to gather feedback. For the category of data acquisition guidance (Q1-Q3), we asked the participants if the designed guidance was easy to understand and follow and the guidance generation was fast. For the category of system experience (Q4-Q8), we asked the participants if the system was helpful and if they felt more confident and more involved when using the system. All questions in these categories were answered on a five-point Likert scale. At the end of the survey, we asked the participants to identify their favorite system and to leave open-ended feedback about the overall experience.
  * Q1: The BiGuide guidance content was easy to understand.
  * Q2: The BiGuide guidance itself was easy to follow.
  * Q3: The BiGuide guidance displayed fast after pressing the "take the image button".
  * Q4: BiGuide was helpful for collecting diverse data.
  * Q5: FreGuide was helpful for collecting diverse data.
  * Q6: BiGuide made me confident in that I was collecting useful data.
  * Q7: FreGuide made me confident in that I was collecting useful data.
  * Q8: BiGuide made me change poses, locations, and angles more frequently.
    
### 5.2. User Study Instructions
During the user study, each participant began by reviewing a set of instructions we have prepared. The instructions for indoor scenario are the same as the instruction for wildlife exhibits scenario. We take the instructions in indoor scenario below for instance.

#### 5.2.1. Intro
The goal of this experiment is to collect diverse data to train a machine-learning model that can detect the class and the bounding box of indoor objects. Seven indoor objects are included in this experiment. They are located in 7 different places in the lab. 

#### 5.2.2. Instructions
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

## 6. <span id="6"> Citation</span>

Please cite the following paper in your publications if the dataset helps your research.

     @inproceedings{Duan24BiGuide,
      title={{BiGuide}: A bi-level data acquisition guidance for object detection on mobile devices },
      author={Duan, Lin and Chen, Ying and Qu, Zhehan and McGrath, Megan and Ehmke, Erin and Gorlatova, Maria},
      booktitle={Proc. IEEE/ACM IPSN},
      year={2024}
    }

## 7. <span id="7"> Acknowledgments</span>

The contributors of the code are [Lin Duan](https://scholar.google.com/citations?user=3KGmyogAAAAJ&hl=en) and [Maria Gorlatova](https://maria.gorlatova.com/bio/). For questions on this repository or the related paper, please contact Lin Duan at ld213 [AT] duke [DOT] edu.

This work was supported in part by NSF grants CSR-1903136, IIS-2231975, and CNS-1908051, NSF CAREER Award IIS-2046072, Meta Research Award, and Defense Advanced Research Projects Agency Young Faculty Award HR0011-24-1-0001. We gratefully acknowledge the contributions of the Duke Lemur Center and the support provided by the Duke Lemur Center's NSF DBI-2012668 Award.
