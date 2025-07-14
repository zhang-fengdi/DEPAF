<img src="readme_assets/logo.png" width="125" align="right">

# DEPAF

## Contents

- [Overview](#overview)
- [Key features](#key-features)
- [Applications](#applications)
- [Representative results](#representative-results)
- [Getting started](#getting-started)
- [License](#license)
- [Citation](#citation)

## Overview

<img src="readme_assets/method_overview.png" width="800" align="middle">

**DE**ep **PA**ttern **F**itting (DEPAF) is a self-contrastive learning-based fluorescence microscopy signal detection framework featuring interference robustness *without* interference modeling, high generalizability, and unsupervised training within a compact, user-friendly design. 
Leveraging the inherent structural regularity of fluorescence microscopy signals, such as point spread functions (PSFs), periodic fluctuations, and specific frequency distributions, DEPAF utilizes repetitive combinations of small data segments containing “patterns of interest” (POIs) to universally represent these diverse signals.
By contrasting the consistency between different augmented views of the input data generated asynchronously via asymmetric model paths, DEPAF learns discriminative features determined solely by interference level rather than specific interference models. These features are embedded into a newly introduced subpixel-level dense representation scheme, ensuring strong discriminability between overlapping signals and thereby substantially improving the detectable signal density, even under conditions of strong noise and severe background interference with unknown types and parameters.

For more details, please see the paper: 
["*Self-contrastive learning enables interference-resilient and generalizable fluorescence microscopy signal detection without interference modeling*".](https://doi.org/10.1101/2025.04.08.645087)

If you want to reproduce our paper results or refer to the preprocessing/postprocessing scripts, please visit the [paper_reproduction](https://github.com/zhang-fengdi/DEPAF/tree/main/paper_reproduction).

## Key features

- **Interference robustness**: Achieves accurate ultra-high-density signal detection (recognizing and localizing signals) even in severe noise and strong fluorescence backgrounds.
- **Unsupervised training**: Trains without requiring any external ground-truth data.
- **High generalizability**: Different POI examples can represent diverse fluorescence microscopy signals, allowing DEPAF to generalize to broader applications.
- **High accessibility**: Adapting it to new tasks requires only providing example signal patterns and adjusting a single sensitivity parameter, with no specialized expertise required.

## Applications

DEPAF excels in various fluorescence microscopy signal detection tasks by using different POI examples, including:

- **2D SMLM reconstruction with calibrated 2D PSFs as POI examples**:
  - Improves temporal resolution from seconds to 76.8 ms while maintaining a spatial resolution of ~30 nm.
  - Enables super-resolution imaging of subtle cytoskeletal dynamics in live cells.

- **3D SMLM reconstruction with calibrated 3D PSFs as POI examples**:
  - Supports blinking densities at least 10 times higher than standard methods.
  - Requires only an average photon count of 10.8% compared to typical STORM data labeled with Alexa647.

- **3D MERFISH spot signal detection with fitted elliptical Gaussian PSFs as POI examples**:
  - Increases the RNA detection rate by 97% compared to the conventional method.
  - Improves the correlation between detected RNA copy numbers and FPKM by 0.12 compared to the conventional method.

- **Two-photon calcium imaging processing with fitted spike signals and standard Gaussian PSFs as POI examples**:
  - Performs denoising, background removal, weak spike detection, and neuron segmentation in a unified pipeline.
  - Outperforms methods of the same unsupervised type with 40% higher precision and 43% higher recall.
  - Matches top-performing supervised methods while reducing the required training data and labels by 2 and 1 orders of magnitude, respectively.

- **Fluorescence background estimation with homogeneous patches as POI examples**:
  - Achieves accurate estimation of *arbitrary inhomogeneous* fluorescence backgrounds.
  - Correlations with ground truths exceeding 0.95 across simulated datasets with various background variation scales and SNR levels.

- **Using more well-designed POI examples:**
  - Unlocks infinite possibilities...

## Representative results

Here we highlight some of DEPAF’s results in addressing challenging imaging scenarios.

### Millisecond-level dynamic 2D SMLM

This video demonstrates DEPAF’s ability to achieve millisecond-level temporal resolution in 2D SMLM, 
enabling observation of minute changes in live cells at the cytoskeletal level.

[![Demo video 1](https://img.youtube.com/vi/ouvD0Bvy2mY/0.jpg)](https://www.youtube.com/watch?v=ouvD0Bvy2mY)

### Denoising with background separation for low-SNR two-photon calcium imaging data

This video demonstrates DEPAF’s ability to denoise two-photon calcium imaging data, effectively separating the backgrounds while preserving neuronal activation signals.

[![Demo video 2](https://img.youtube.com/vi/9urDrTqq04I/0.jpg)](https://www.youtube.com/watch?v=9urDrTqq04I)

## Getting started

### Recommended environment

- **MATLAB:** Version 2022a or newer
- **Operating System:** Windows 10 or 11
- **Memory:** 32 GB RAM (additional memory may be required for large datasets or complex tasks)
- **GPU:** NVIDIA GPU with CUDA compatibility (recommended for accelerated processing)

### Installation

1. Download the package to a local folder (e.g. ~/DEPAF/) by running:

    ```bash
    git clone https://github.com/zhang-fengdi/DEPAF.git
    ```

2. Run MATLAB and navigate to the folder (~/DEPAF/)

3. Add the DEPAF source code to the MATLAB search path:

    ```matlab
    addpath(genpath('DEPAF_src_dev'));
    ```

4. Use the example below to train a model (install any required MATLAB toolboxes according to runtime prompts):

    ```matlab
    % Parameters:
    dataPath = ''; % Path to image data
    POIPath = ''; % Path to POI image data
    lambda = 0.0001; % Regularization intensity coefficient

    DEPAFTrain(dataPath, POIPath, lambda);
    ```

    **Note:** `lambda` is a parameter that controls the sensitivity of signal detection and requires tuning depending on your specific task. A higher `lambda` value makes signal detection stricter, increasing noise resistance but potentially losing signal, while a lower value may retain more signal but be less resistant to noise. You can check the parameter settings in [paper_reproduction](https://github.com/zhang-fengdi/DEPAF/tree/main/paper_reproduction) for reference. `dataPath` and `POIPath` must refer to either a `.mat` or `.tif` image file. If a `.mat` file is provided, it must contain a single matrix variable with dimensions of “image height × image width × number of images” to ensure proper data loading and processing.
   
    To further customize the training process, see the full version below:

   <details style="margin-bottom:1em"> <summary>Show full version</summary>
     
    ```matlab
    % Required Parameters:
    dataPath = ''; % Path to image data
    POIPath = ''; % Path to POI image data
    lambda = 0.0001; % Regularization intensity coefficient

    % Optional Parameters:
    % Data loading related parameters:
    trainIdxRange = 1:8; % Training set patch sampling index range
    valIdxRange = 9:10; % Validation set patch sampling index range

    % Model saving related parameters:
    modelSavePath = '.\'; % Path to save the model

    % Preprocessing related parameters:
    upsamplRatio = [1.5 1.5 1]; % Upsampling factor
    interpMethod = 'spline'; % Interpolation method, options: 'spline', 'linear', 'nearest', 'cubic'

    % Model structure parameters:
    encoderDepth = 2; % Depth of the U-net encoder (total depth approximately doubled)

    % Model training parameters:
    patchSize = [256 256]; % Size of patches
    trainPatchNum = 512; % Number of training set patches
    valPatchNum = 64; % Number of validation set patches
    learningRate = 1e-3; % Learning rate
    minLR = 1e-4; % Lower limit for learning rate decay
    miniBatchSize = 8; % Batch size per iteration
    maxEpochs = 1000; % Maximum number of training epochs
    valFreq = 20; % Validation frequency (validate every valFreq batches)
    maxPatience = 20; % Early stopping patience (stop training if validation loss does not decrease for this many epochs)
    learnBG = true; % Whether to learn background
    verbose = true; % Display processing as images
    useGPU = true; % Use GPU

    % Bayesian estimation optimal threshold search parameters:
    useParallel = true; % Use parallel computation
    parNum = 6; % Number of parallel pool workers
    patchNumForThreshSearch = 512; % Number of samples for optimal segmentation threshold search

    DEPAFTrain(dataPath, POIPath, lambda, ...
        'trainIdxRange', trainIdxRange, ...
        'valIdxRange', valIdxRange, ...
        'upsamplRatio', upsamplRatio, ...
        'interpMethod', interpMethod, ...
        'patchSize', patchSize, ...
        'trainPatchNum', trainPatchNum, ...
        'valPatchNum', valPatchNum, ...
        'encoderDepth', encoderDepth, ...
        'learningRate', learningRate, ...
        'minLR', minLR, ...
        'miniBatchSize', miniBatchSize, ...
        'maxEpochs', maxEpochs, ...
        'valFreq', valFreq, ...
        'maxPatience', maxPatience, ...
        'learnBG', learnBG, ...
        'verbose', verbose, ...
        'useGPU', useGPU, ...
        'useParallel', useParallel, ...
        'parNum', parNum, ...
        'patchNumForThreshSearch', patchNumForThreshSearch, ...
        'modelSavePath', modelSavePath);
    ```
    
    </details>

6. After training the model, perform predictions using the example below (install any required MATLAB toolboxes according to runtime prompts):

    ```matlab
    % Parameters:
    modelPath = ''; % Path to prediction model
    dataPath = ''; % Path to prediction data
    batchSize = 64; % Batch size for a single prediction
    patchSize = [64 64]; % Patch size for prediction (must be a power of 2)
    patchStride = [32 32]; % Patch stride for prediction
    resSavePath = '.\'; % Path to save prediction results
    outputISyn = false; % Output noiseless synthetic images
    outputBG = false; % Output background images
    useGPU = true; % Use GPU

    DEPAFPred(modelPath, dataPath, ...
        'batchSize', batchSize, ...
        'patchSize', patchSize, ...
        'patchStride', patchStride, ...
        'resSavePath', resSavePath, ...
        'outputISyn', outputISyn, ...
        'outputBG', outputBG, ...
        'useGPU', useGPU);
    ```

    **Note:** `dataPath` and `POIPath` must refer to either a `.mat` or `.tif` image file. If a `.mat` file is provided, it must contain a single matrix variable with dimensions formatted as “image height × image width × number of images” to ensure proper data loading and processing.

## License

DEPAF is released under the [GPL-3.0 License](https://github.com/zhang-fengdi/DEPAF/blob/main/LICENSE).

## Citation

We kindly ask you to cite the following reference if you use this code in your research: 

Zhang, F. et al. Self-contrastive learning enables interference-resilient and generalizable fluorescence microscopy signal detection without interference modeling. *bioRxiv* (2025). [https://doi.org/10.1101/2025.04.08.645087](https://doi.org/10.1101/2025.04.08.645087)

```
@article{zhang2025depaf,
  title = {Self-contrastive learning enables interference-resilient and generalizable fluorescence microscopy signal detection without interference modeling},
  author = {Zhang, Fengdi and Huang, Ruqi and Xin, Meiqian and Meng, Haoran and Gao, Danheng and Fu, Ying and Gao, Juntao and Ji, Xiangyang},
  journal = {bioRxiv},
  DOI = {10.1101/2025.04.08.645087},
  year = {2025}
}
```
