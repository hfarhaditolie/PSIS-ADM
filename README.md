<div align="center">

# Promptable Sonar Image Segmentation for Distance Measurement using SAM
[**Hamidreza Farhadi Tolie**](https://scholar.google.com/citations?user=nzCbjWIAAAAJ&hl=en&authuser=1)<sup>a, b</sup> · [**Jinchang Ren**](https://scholar.google.co.uk/citations?user=Vsx9P-gAAAAJ&hl=en)<sup>a, b</sup> · [**Md Junayed Hasan**](https://scholar.google.com/citations?user=6dp1PZAAAAAJ&hl=en)<sup>a, b</sup> · [**Somasundar Kannan**](https://scholar.google.com/citations?user=-GDIrlMAAAAJ&hl=en)<sup>b</sup> · [**Nazila Fough**](https://rgu-repository.worktribe.com/person/74551/nazila-fough)<sup>b</sup>

<sup>a</sup> National Subsea Centre, Robert Gordon University, UK

<sup>b</sup> School of Computing, Engineering, and Technology, Robert Gordon University, UK

<hr>

<a href='https://drive.google.com/file/d/12jCgB8_48m2ancWF7SnVBnK4z1e9iNaU/view?usp=sharing'><img src='https://img.shields.io/badge/%20PSIS%20-%20Paper?label=Poster&labelColor=(255%2C0%2C0)&color=red' alt='Will be available soon!'></a>
<br>

</div>

This repository contains collected sonar data during the experiments with their associated ground-truth distance measurements.


## Abstract

> The subsea environment presents numerous challenges for robotic vision, including non-uniform light attenuation, backscattering, floating particles, and low-light conditions, which significantly degrade underwater images. This degradation impacts robotic operations that heavily rely on environmental feedback. However, these limitations can be mitigated using sonar imaging, which employs sound pulses instead of light. In this paper, we explore the use of small, affordable sonar devices for automatic target object localization and distance measurement. Specifically, we propose using a promptable image segmentation method to identify target objects within sonar images, leveraging its ability to identify connected components without requiring labeled datasets. Through laboratory experiments, we analyzed the usability of the Ping360 single-beam sonar and verified the effectiveness of our approach in the automatic identification and distance measurement of objects made from various materials.
---

<p align="center">
  <img src="https://hamidrezafarhadi.com/publications/Images/PSIS/framework.PNG" alt="Framework Overview" width="500">
  <br>
  <em>Figure 1: General framework of the proposed methodology using SAM</em>
</p>

<p align="center">
  <img src="https://hamidrezafarhadi.com/publications/Images/PSIS/watertank.PNG" alt="Watertank Setup" width="500">
  <br>
  <em>Figure 2: A schematic diagram of the testing environment within a water tank</em>
</p>

## Usage
To utilize the DICAM method for training, please follow these steps:

1. Download the Segment-Anything model and checkpoints from its respective source.
   - [Segment-Anything](https://github.com/facebookresearch/segment-anything/tree/main?tab=readme-ov-file#installation)
   
2. Run the _offline_detector.py_ script to generate the SONAR image from the shared data and then produce the detection result.

3. Run the _acquisition_detection.py_ script to have real-time data acquistion and target detection using Ping360 and Segment-Anything, respectively.
## Dependencies
```bash
Ping-Python -> https://github.com/bluerobotics/ping-python
```
## Citation
```bash
@INPROCEEDINGS{10765703,
  author={Tolie, Hamidreza Farhadi and Ren, Jinchang and Hasan, Md. Junayed and Kannan, Somasundar and Fough, Nazila},
  booktitle={2024 IEEE International Workshop on Metrology for the Sea; Learning to Measure Sea Health Parameters (MetroSea)}, 
  title={Promptable Sonar Image Segmentation for Distance Measurement Using SAM}, 
  year={2024},
  volume={},
  number={},
  pages={229-233},
  keywords={Water;Image segmentation;Accuracy;Thresholding (Imaging);Sonar measurements;Storage management;Sea measurements;Distance measurement;Object recognition;Usability;Sonar image segmentation;distance measurement;Ping360;single-beam sonar},
  doi={10.1109/MetroSea62823.2024.10765703}}
```
## Feedback
If you have any enquires or feedback, please do not hesitate to contact us via @(h.farhadi-tolie@rgu.ac.uk, h.farhaditolie@gmail.com)

## Acknowledgement
We extend our gratitude to the creators of Segment-Anything for generously sharing their source code, which can be accessed [here](https://github.com/facebookresearch/segment-anything). This has greatly simplified the process of loading images from individual datasets.

## License
This project is licensed under the [MIT License](LICENSE).
