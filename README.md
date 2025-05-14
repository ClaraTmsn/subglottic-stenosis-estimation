# Subglottic Stenosis Estimation
**Related publication:** Tomasini, C., Rodríguez-Puigvert, J., Polanco, D., Viñuales, M., Riazuelo, L., Murillo, A.C. (2025). "Automated vision-based assistance tools in bronchoscopy: stenosis severity estimation.", International Journal of Computer Assisted Radiology and Surgery 
[[arxiv]](https://arxiv.org/pdf/2505.05136) [[Project page]](https://sites.google.com/unizar.es/subglottic-stenosis-estimation/home)

# License 
Step 1 of Subglottic Stenosis Estimation pipeline (Keyframe Selection step) is released under AGPLv3 license.

# Pre-requisites
The software has been tested on Ubuntu 20.04 and uses Python. Required 3.X.
### Required packages:
* Numpy 1.14.0
* Opencv 4.11.0
* Matplotlib
* ffmpeg 4.2.7
  
# Proposed Pipeline
Our proposed pipeline to estimate stenosis severity consists of two steps: 1) Segmentation, Tracking and Keyframe Selection 2) 3D Reconstruction and Stenosis Estimation. Stenosis is measured in step 2 at the keyframe selected by step 1.
![results](/images/pipeline_stenosis.png)

# Data
Our pipeline is evaluated on Subglottic Stenosis (SGS) Dataset, available [here](https://sites.google.com/unizar.es/subglottic-stenosis-estimation/home). Our dataset contains 16 bronchoscopy videos from 11 patients, with the following characteristics:
![results](/images/sgs_dataset.png)
Frames can be extracted from each video using ffmpeg at framerate 15 fps:  
```ffmpeg -i stenosis_1.mp4 -filter:v fps=15 frame_%05d.png```
# Step 1: Keyframe Selection
