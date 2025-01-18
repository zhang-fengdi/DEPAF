<img src="readme_assets/logo.png" width="125" align="right">

# DEPAF

## Contents

- [DEPAF](#depaf)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Key features](#key-features)
  - [Code Availability](#code-availability)

## Overview

**DE**ep **PA**ttern **F**itting (DEPAF) is a self-supervised, deep learning-based fitting framework designed to detect and localize ultra-high-density fluorescence microscopy signals. Specifically, DEPAF introduces patterns of interest (POIs) to represent various fluorescence microscopy signals and enables ultra-high-density POI localization even under severe noise and intense fluorescence backgrounds by performing parallel POI fitting and global denoising within a large receptive field. This approach effectively advances the detectable density of fluorescence microscopy signals to ultra-high levels under strong interference. By simply replacing the POI examples and adjusting a sensitivity-related hyperparameter, DEPAF boosts the performance of various tasks and addresses several long-standing challenges.

## Key features

- **Ultra-resolving detection**: Achieves accurate ultra-high-density signal detection and localization even in severe noise and strong fluorescence backgrounds.
- **Self-supervised learning**: Trains without the need for any manually labeled or simulated ground-truth data.
- **Denoised data output**: Thanks to the accurate estimation of POI amplitudes and positions, DEPAF can output high-quality denoised data.
- **User-friendly**: Works by simply providing the POI examples and adjusting a single hyperparameter related to detection sensitivity.

## Code Availability

The source code for DEPAF is currently under preparation and will be made available once the associated research work is officially published. 

In the meantime, if you have specific questions or require access for academic collaboration, please feel free to contact us at [zhangfengdi.thu@gmail.com](mailto:zhangfengdi.thu@gmail.com).