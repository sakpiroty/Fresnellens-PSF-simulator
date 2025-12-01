
# Differentiable PSF Simulator for Coded Fresnel Lens

This optical simulator is for a shadowless projector using coded large-aperture Fresnel lens.
The PSF of the Fresnel-lens based projector is simulated in differentiable manner, and the coded aperture of the lens is optimized in terms of deblurring and shadow suppression depending on the simulated PSF.

<img width="1122" height="920" alt="image" src="https://github.com/user-attachments/assets/72f289c9-933d-44ce-9669-f10be77637a4" />

The entire process of the aperture optimization.
<img width="2462" height="1204" alt="image" src="https://github.com/user-attachments/assets/cdc1c6ee-6705-45bf-80c7-3804eeb88ae8" />

# Requirements
Python 3
Pillow 
Torch 

# How to Use
・opt_3D.py: optimizing the coded aperture in specific optical configuration of the coded large-aperture projector using Fresnel lens.
・PSF_in_depth.py: PSF simulation using a certain mask (totally-open, random, optimized etc.).
