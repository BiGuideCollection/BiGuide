# BiGuide: A Bi-level Data Acquisition Guidance for Object Detection on Mobile Devices

This repository contains download links and the introduction of our collected indoor and lemur datasets, as well as the code for IPSN 2024 paper: ["BiGuide: A Bi-level Data Acquisition Guidance for Object Detection on Mobile Devices"]() by [Lin Duan](https://scholar.google.com/citations?user=3KGmyogAAAAJ&hl=en), [Ying Chen](https://scholar.google.com/citations?hl=en&user=aoMpAKoAAAAJ), Zhehan Qu, Megan McGrath, Erin Ehmke, and [Maria Gorlatova](https://maria.gorlatova.com/bio/). 

The BiGuide implementation is based on [mmyolo](https://github.com/open-mmlab/mmyolo).

**Outline**:
* [I. Prerequisites](#1)
* [II. Post-experiment Survey Questions](#2)
* [III. User Study Instructions](#3)
* [IV. Dataset](#4)
* [V. Running BiGuide on Client and Server Devices](#5)


## I. <span id="1"> Prerequisites</span>
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

## II. <span id="2">Post-experiment Survey Questions</span>
We assembled a set of questions in different categories for the post-experiment survey to gather feedback. For the category of data acquisition guidance (Q1-Q3), we asked the participants if the designed guidance was easy to understand and follow and the guidance generation was fast. For the category of system experience (Q4-Q8), we asked the participants if the system was helpful and if they felt more confident and more involved when using the system. All questions in these categories were answered on a five-point Likert scale. At the end of the survey, we asked the participants to identify their favorite system and to leave open-ended feedback about the overall experience.
  * Q1: The BiGuide guidance content was easy to understand.
  * Q2: The BiGuide guidance itself was easy to follow.
  * Q3: The BiGuide guidance displayed fast after pressing the "take the image button".
  * Q4: BiGuide was helpful for collecting diverse data.
  * Q5: FreGuide was helpful for collecting diverse data.
  * Q6: BiGuide made me confident in that I was collecting useful data.
  * Q7: FreGuide made me confident in that I was collecting useful data.
  * Q8: BiGuide made me change poses, locations, and angles more frequently.
    
## III. <span id="3">User Study Instructions</span>
During the user study, each participant began by reviewing a set of instructions we have prepared. The instructions for indoor scenario are the same as the instruction for wildlife exhibits scenario. We take the instructions in indoor scenario below for instance.
### III-I. Intro
The goal of this experiment is to collect diverse data to train a machine-learning model that can detect the class and the bounding box of indoor objects. Seven indoor objects are included in this experiment. They are located in 7 different places in the lab. 
### III-II. Instructions
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

## IV. <span id="4">Dataset</span>
We share the data collected using a commodity Google Pixel 3 XL smartphone during the user study in the **indoor scenario** and **wildlife exhibits scenario**. The detailed information about the collected datasets is presented below.

### IV-I. Indoor Scenario
We set up the indoor scenario in a typical office environment. **10 users** were guided to collect **20 images for each object** in this environment. We included **seven object classes**: mobile phone, scissors, ball, tin can, light bulb, mug, and remote control. These objects were placed in **seven distinct locations** within a controlled environment. Users moved around different locations to collect the images of the objects. 14 images have been removed to address privacy concerns. The details are summarized in the table below:

<table border="0">
    <tr>
        <td>Number of users</td><td>10</td>
    </tr>
    <tr>
        <td>Number of object classes</td><td>7</td>
    </tr>
    <tr>
        <td>Number of images per class</td><td>20</td>
    </tr>
    <tr>
        <td><b>Total images</b></td><td>10 x 7 x 20 - 14 = 1,388</td>
    </tr>
</table>

#### Examples of images in the indoor dataset:
<p align="center">
  <img src="https://github.com/BiGuideCollection/BiGuide/assets/138166113/7abc914f-ea9e-4a2a-9c44-9c7d1d63c6b9" width="500" alt="Alt text" title="Figure 1. Example images of 7 objects positioned in 7 locations in the indoor scenario. These objects were placed in a controlled environment with stable lighting conditions.">
</p>
<p align="center">
    <em>Figure 1. Example images of 7 objects positioned in 7 locations in the indoor scenario. These objects were placed in a controlled environment with stable lighting conditions.</em>
</p>

### IV-II. Wildlife Exhibit Scenario
We set up the wildlife exhibit scenario in the Duke Lemur Center. This scenario involves outdoor scenes with dynamic objects, specifically lemurs. **10 users** were tasked with capturing **20 images for each lemur species**. **3 lemur species** were showcased in the center: blue-eyed black lemur, ring-tailed lemur, and red ruffed lemur. Different lemur species were housed in distinct enclosures, requiring users to move between these separate areas. Users’ visits were scheduled at different times on seven different days, aligning with the center’s general tour schedule. This led to users encountering different weather conditions, including sunny and heavily rainy days. On warm, sunny days, the lemurs were more active, engaging in activities like climbing and exploring; on cold, rainy days, the lemurs tended to gather and rest inside their cages. Compared to the images collected in the indoor scenario, the wildlife images present greater complexity and detection challenges due to the lemurs’ varied poses and sizes, occlusion from cages, and unstable lighting conditions. 20 images have been removed to address privacy concerns. The details are summarized in the table below:

<table border="0">
    <tr>
        <td>Number of users</td><td>10</td>
    </tr>
    <tr>
        <td>Number of object classes</td><td>3</td>
    </tr>
    <tr>
        <td>Number of images per class</td><td>20</td>
    </tr>
    <tr>
        <td><b>Total images</b></td><td>10 x 3 x 20 - 20 = 580</td>
    </tr>
</table>

#### Examples of images in the wildlife exhibit dataset:
<p align="center">
  <img src="https://github.com/BiGuideCollection/BiGuide/assets/138166113/6d1495ec-116f-48cc-945d-adf97526ea6d" width="500" alt="Alt text" title="Figure 2.Example images of 3 lemur species enclosed in 3 distinct exhibits in the wildlife exhibits scenario. Images obtained in this scenario are more complex due to lemurs’ varying poses and sizes, as well as the diverse backgrounds.">
</p>
<p align="center">
    <em>Figure 2.Example images of 3 lemur species enclosed in 3 distinct exhibits in the wildlife exhibits scenario. Images obtained in this scenario are more complex due to lemurs’ varying poses and sizes, as well as the diverse backgrounds.</em>
</p>

### IV-III. Download Indoor and Wildlife Exhibit Datasets
+ The indoor dataset can be downloaded via https://duke.box.com/s/vvblvq6mp8i8gbt6ik2l6rxrscfaukci
+ The wildlife exhibit dataset can be downloaded via https://duke.box.com/s/5iu2er13s0kbmr79e8x81bcv6d160ku5

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
    -target_2023_MM_DD_hh_mm_ss_XXXXXX.jpg
    -...
  -class_with_id.txt
```
The images are named by the time when they were captured.

### IV-IV. Data Distribution Comparision:
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

## V. <span id="5">Running BiGuide on Client and Server Devices</span>
We take the indoor scenario as an example.
### V-I. Running BiGuide on Server
Change the path in UserStudy_BiGuide_indoor.py to your own path. Then run the command below.
```
cd tools/
uvicorn UserStudy_BiGuide_indoor:app --reload --host xxx.xxx.x.x --port xxxx
```
Replace ```--host xxx.xxx.x.x --port xxxx``` with your own IP address and port number.

### V-II. Running BiGuide on Mobile Devices
When the code on the server is running, you can build the mobile app through Unity with the code in "APP.zip". Then, enjoy your data collection!

## <span id="6">Citation</span>
Please cite the following paper in your publications if the dataset helps your research.

     @inproceedings{Duan24BiGuide,
      title={{BiGuide}: A bi-level data acquisition guidance for object detection on mobile devices },
      author={Duan, Lin and Chen, Ying and Qu, Zhehan and McGrath, Megan and Ehmke, Erin and Gorlatova, Maria},
      booktitle={Proceedings of the 23rd ACM/IEEE Conference on Information Processing in Sensor Networks},
      year={2024}
    }

## <span id="7">Acknowledgments</span>

The authors of this project are [Lin Duan](https://scholar.google.com/citations?user=3KGmyogAAAAJ&hl=en), [Ying Chen](https://scholar.google.com/citations?hl=en&user=aoMpAKoAAAAJ), Zhehan Qu, Megan McGrath, Erin Ehmke, and [Maria Gorlatova](https://maria.gorlatova.com/bio/). This work was done in the [Intelligent Interactive Internet of Things Lab](https://maria.gorlatova.com/) at [Duke University](https://www.duke.edu/).

Contact Information of the contributors: 

* lin.duan AT duke.edu
* ying.chen151 AT me.com
* zhehan.qu AT duke.edu
* maria.gorlatova AT duke.edu

We thank Ashley Kwon for her contributions to the project. This work was supported in part by NSF grants CSR-1903136, IIS-2231975, and CNS-1908051, NSF CAREER Award IIS-2046072, Meta Research Award, and Defense Advanced Research Projects Agency Young Faculty Award HR0011-24-1-0001. This paper has been approved for public release; distribution is unlimited. The contents of the paper do not necessarily reflect the position or the policy of the Defense Advanced Research Projects Agency. No official endorsement should be inferred. This is Duke Lemur Center publication \#1586. We gratefully acknowledge the contributions of the Duke Lemur Center and the support provided by the Duke Lemur Center's NSF DBI-2012668 Award.
