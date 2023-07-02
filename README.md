This repository contains the code for the **SenSys 2023** submitted paper: *"[BiGuide: A Bi-level Data Acquisition Guidance for Object Detection on Mobile Devices]"*. 

The BiGuide implementation is based on [mmyolo](https://github.com/open-mmlab/mmyolo).

## Outline
* [I. Prerequisites](#1)
* [II. Post-experiment Survey Questions](#2)
* [III. User Study Instruction](#3)
* [IV. Running BiGuide on Client and Server Devices](#4)


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
  * 
#### <span id="3">III. User Study Instruction
Follow the instructions in (https://github.com/open-mmlab/mmyolo).

#### <span id="4">IV. Running BiGuide on Client and Server Devices
##### 1. Running BiGuide on Server
```
 cd AdaptSLAM/Edge-assisted AdaptSLAM
 chmod +x build.sh
 ./build.sh
 ```
 ##### 2. Running BiGuide on Mobile Devices
```
./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml V102FileDirectory ./Examples/Monocular/EuRoC_TimeStamps/V102.txt dataset-V102_mono
```
