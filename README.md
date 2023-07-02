This repository contains the code for the **SenSys 2023** submitted paper: *"[BiGuide: A Bi-level Data Acquisition Guidance for Object Detection on Mobile Devices]"*. 

The BiGuide implementation is based on [mmyolo](https://github.com/open-mmlab/mmyolo)

## Outline
* [I. Prerequisites](#1)
* [II. Post-experiment Survey Questions](#2)
* [III. User Study Instruction](#3)
* [IV. Running BiGuide on Client and Server Devices](#4)


#### <span id="1">I. Prerequisites
Setup the [mmyolo](https://github.com/open-mmlab/mmyolo) prerequisites.

#### <span id="2">II. Post-experiment Survey Questions
  * Dell XPS 8930 desktop with Intel (R) Core (TM) i7-9700K CPU@3.6GHz and NVIDIA GTX 1080 GPU, and a Lenovo Legion 5 laptop (with an AMD Ryzen 7 4800H CPU and an NVIDIA
GTX 1660 Ti GPU) using a virtual machine with 4-core CPUs and 8GB of RAM.
  * Ubuntu 18.04LTS.
  * OpenCV 3.4.2.
  * Eigen3 3.2.10.
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
