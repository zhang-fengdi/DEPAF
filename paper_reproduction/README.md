# Paper Reproduction

To reproduce the paper results, follow these steps:

1. **Download datasets and models**:

      | Part | Project name         | Downloads                                                                                                     |
      |:----:|:--------------------:|:-------------------------------------------------------------------------------------------------------------:|
      | 1    | 2D SMLM benchmarking | [Dataset](https://zenodo.org/your-zenodo-link)             \| [Models](https://zenodo.org/your-zenodo-link)|
      | 2    | Dynamic 2D SMLM      | [Dataset](https://zenodo.org/your-zenodo-link)             \| [Models](https://zenodo.org/your-zenodo-link)|
      | 3    | 3D SMLM benchmarking | [Dataset](https://zenodo.org/your-zenodo-link)<sup>a</sup> \| [Models](https://zenodo.org/your-zenodo-link)|
      | 4    | MERFISH analysis     | Dataset<sup>b</sup>                                        \| [Models](https://zenodo.org/your-zenodo-link)|
      | 5    | Denoising estimation | [Dataset](https://zenodo.org/your-zenodo-link)             \| [Models](https://zenodo.org/your-zenodo-link)|
      | 6    | 2PCI processing      | [Dataset](https://zenodo.org/your-zenodo-link)<sup>c</sup> \| [Models](https://zenodo.org/your-zenodo-link)|
      | 7    | Background estimation| [Dataset](https://zenodo.org/your-zenodo-link)             \| [Models](https://zenodo.org/your-zenodo-link)|                                                                    |

      <sup>a</sup> The dataset for **3D SMLM benchmarking** can also be downloaded from [here](https://srm.epfl.ch/srm/dataset/challenge-3D-simulation/index.html).
      
      <sup>b</sup> The dataset for **MERFISH analysis** is available upon request from the authors of ref. [[1]](#references).

      <sup>c</sup> The dataset *CaImAn Video J123* for **2PCI processing** can also be downloaded from [here](https://zenodo.org/records/1659149).

2. **Organize downloaded data**:
   - Place the downloaded datasets into the corresponding subfolders within the `datasets` directory.
   - Place the downloaded models into the corresponding subfolders within the `models` directory.

3. **Run reproduction code**:
   - Run MATLAB, open the subfolder corresponding to the desired part to reproduce, and navigate to the `code` directory.
   - Execute the reproduction scripts.
   
   **Note**: To ensure precise reproduction of the paper's results, it is recommended to use the downloaded models directly for prediction and subsequent processing. Re-training the models may lead to results that differ from those reported in the paper.

## References

[1] Lu Y., Liu M., Yang J., et al. Spatial transcriptome profiling by MERFISH reveals fetal liver hematopoietic stem cell niche architecture. *Cell Discovery* (2021). [Link](https://doi.org/10.1038/s41421-021-00266-1)

## Citation

If you use all the models provided here or any of the datasets for 2D SMLM benchmarking, dynamic 2D SMLM, denoising estimation, or background estimation, please cite the following paper:

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