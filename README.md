<p align="center">

  <h1 align="center">Endo-4DGX: Robust Endoscopic Scene Reconstruction and Illumination Correction with Gaussian Splatting</h1>

  <h2 align="center">MICCAI 2025</h2>
  <p align="center">
    <a href="https://github.com/lastbasket"><strong>Yiming Huang*</strong></a>,
    <a href="https://longbai-cuhk.github.io/"><strong>Long Bai*</strong></a>,
    <a href="https://beileicui.github.io/"><strong>Beilei Cui*</strong></a>,
    <strong>Yanheng Li</strong>,
    <a href="https://davismeee.github.io/"><strong>Tong Chen</strong></a>,
    <br>
    <strong>Jie Wang</strong>,
    <strong>Jinlin Wu</strong>,
    <strong>Zhen Lei</strong>,
    <strong>Hongbin Liu</strong>,
    <a href="https://www.ee.cuhk.edu.hk/ren/"><strong>Hongliang Ren</strong></a>
  </p>
  <h3 align="center"> || <a href="https://arxiv.org/abs/2506.23308">Paper</a> || <a href="https://lastbasket.github.io/MICCAI-2025-Endo-4DGX/">Project Page</a> || </h3>
  <div align="center"></div>
</p> 
<p align="center">
  <a href="https://lastbasket.github.io/MICCAI-2025-Endo-4DGX/">
    <img src="./figs/fig2_3-1.png" alt="Logo" width="90%">
  </a>
</p>

## TODO
- [ ] Evaluation
- [ ] Dataset Preprocessing

## Environment
1. Install the CUDA toolkit on ubuntu from [Download link](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu), and then:
```shell
export PATH=/usr/local/cuda-11.7/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7
```
2. Install the Python environment
```bash
git clone https://github.com/lastbasket/Endo-4DGX
cd Endo-4DGX
conda create -n endo4dgx python=3.7 
conda activate endo4dgx

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```
## Datasets and Pre-trained Checkpoints
1. We have the processed version of EndoNeRF-EC and StereoMIS (P1_1, P1_2) datasets with depth maps. Download the datasets from the [Download Link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155209042_link_cuhk_edu_hk/EpzElOs7fxpLv6dQ-VFhUzkBbWHbL-LZvugB9P452ZkBvw?e=XUhqcG), unzip to the following structure:
```
├── data
│   ├── endonerfec
│   |   ├── pulling_soft_tissues
│   |   |   ├── depth_dam_adjusted
│   |   |   ├── images_mix
│   |   |   ├── images_mix_adjusted
│   |   |   ├── masks
│   |   |   ├── poses_bounds.npy
│   |   ├── cutting_tissues_twice
│   |   ├── ...
```

## Training
```bash
bash train.sh
```

## Rendering
```bash
bash render.sh
```

## Related Works
Welcome to follow our related works:
- [SurgTPGS](https://lastbasket.github.io/MICCAI-2025-SurgTPGS/): Vison-Language Surgical 3D Scene Understanding
- [Endo2DTAM](https://github.com/lastbasket/Endo-2DTAM): Gaussian Splatting SLAM for Endoscopic Scene
- [Endo-4DGS](https://github.com/lastbasket/Endo-4DGS): Monocular Endoscopic Scene Reconstruction with Gaussian Splatting

## Citation
```
@misc{huang2025endo4dgxrobustendoscopicscene,
      title={Endo-4DGX: Robust Endoscopic Scene Reconstruction and Illumination Correction with Gaussian Splatting}, 
      author={Yiming Huang and Long Bai and Beilei Cui and Yanheng Li and Tong Chen and Jie Wang and Jinlin Wu and Zhen Lei and Hongbin Liu and Hongliang Ren},
      year={2025},
      eprint={2506.23308},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23308}, 
}
```
<p align="center">
