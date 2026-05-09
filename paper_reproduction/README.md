# Paper Reproduction

To reproduce the paper results, follow these steps:

1. **Download datasets and models**:

      <table>
        <thead>
          <tr>
            <th align="center"><div align="center">Part</div></th>
            <th align="center" colspan="2"><div align="center">Project name</div></th>
            <th align="center"><div align="center">Downloads</div></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td align="center">1</td>
            <td align="center" colspan="2">2D SMLM benchmarking</td>
            <td align="center"><a href="https://zenodo.org/records/20093813/files/part1_2D_SMLM_benchmarking_datasets.zip?download=1">Dataset</a> | <a href="https://zenodo.org/records/20093813/files/part1_2D_SMLM_benchmarking_models.zip?download=1">Models</a></td>
          </tr>
          <tr>
            <td align="center">2</td>
            <td align="center" colspan="2">Dynamic 2D SMLM</td>
            <td align="center"><a href="https://zenodo.org/records/20093813/files/part2_dynamic_2D_SMLM_datasets.zip?download=1">Dataset</a> | <a href="https://zenodo.org/records/20093813/files/part2_dynamic_2D_SMLM_models.zip?download=1">Models</a></td>
          </tr>
          <tr>
            <td align="center">3</td>
            <td align="center" colspan="2">3D SMLM benchmarking</td>
            <td align="center"><a href="https://zenodo.org/records/20093813/files/part3_3D_SMLM_benchmarking_datasets.zip?download=1">Dataset</a><sup>a</sup> | <a href="https://zenodo.org/records/20093813/files/part3_3D_SMLM_benchmarking_models.zip?download=1">Models</a></td>
          </tr>
          <tr>
            <td align="center" rowspan="2">4</td>
            <td align="center" rowspan="2">MERFISH analysis</td>
            <td align="center">Fetal liver</td>
            <td align="center">Dataset<sup>b</sup> | <a href="https://zenodo.org/records/20093813/files/part4_MERFISH_analysis_fetal_liver_models.zip?download=1">Models</a></td>
          </tr>
          <tr>
            <td align="center">Mouse colon</td>
            <td align="center"><a href="https://zenodo.org/records/20093813/files/part4_MERFISH_analysis_mouse_colon_datasets.zip?download=1">Dataset</a><sup>c</sup> | <a href="https://zenodo.org/records/20093813/files/part4_MERFISH_analysis_mouse_colon_models.zip?download=1">Models</a></td>
          </tr>
          <tr>
            <td align="center">5</td>
            <td align="center" colspan="2">Denoising estimation</td>
            <td align="center"><a href="https://zenodo.org/records/20093813/files/part5_denoising_estimation_datasets.zip?download=1">Dataset</a> | <a href="https://zenodo.org/records/20093813/files/part5_denoising_estimation_models.zip?download=1">Models</a></td>
          </tr>
          <tr>
            <td align="center">6</td>
            <td align="center" colspan="2">2PCI processing</td>
            <td align="center"><a href="https://zenodo.org/records/20093813/files/part6_2PCI_processing_datasets.zip?download=1">Dataset</a><sup>d</sup> | <a href="https://zenodo.org/records/20093813/files/part6_2PCI_processing_models.zip?download=1">Models</a></td>
          </tr>
          <tr>
            <td align="center">7</td>
            <td align="center" colspan="2">Background estimation</td>
            <td align="center"><a href="https://zenodo.org/records/20093813/files/part7_background_estimation_datasets.zip?download=1">Dataset</a> | <a href="https://zenodo.org/records/20093813/files/part7_background_estimation_models.zip?download=1">Models</a></td>
          </tr>
          <tr>
            <td align="center">8</td>
            <td align="center" colspan="2">Spike inference</td>
            <td align="center"><a href="https://zenodo.org/records/20093813/files/part8_spike_inference_datasets.zip?download=1">Dataset</a> | <a href="https://zenodo.org/records/20093813/files/part8_spike_inference_models.zip?download=1">Models</a></td>
          </tr>
        </tbody>
      </table>

      <sup>a</sup> The dataset for **3D SMLM benchmarking** can also be downloaded from [here](https://srm.epfl.ch/srm/dataset/challenge-3D-simulation/index.html).

      <sup>b</sup> The dataset for **fetal liver MERFISH analysis** is available upon request from the authors of ref. [[1]](#references).

      <sup>c</sup> The dataset for **mouse colon MERFISH analysis** can also be downloaded from [here](https://datadryad.org/dataset/doi:10.5061/dryad.p8cz8wb1m).

      <sup>d</sup> The dataset *CaImAn Video J123* for **2PCI processing** can also be downloaded from [here](https://zenodo.org/records/1659149).

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
