# 🐳 MObI: Multimodal Object Inpainting Using Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2501.03173-b31b1b.svg)](https://arxiv.org/abs/2501.03173)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/pdf/2501.03173)
[![CVPR Workshop](https://img.shields.io/badge/CVPR_Workshop-DDADS-green)](https://agents4ad.github.io/)

Official implementation of "MObI: Multimodal Object Inpainting Using Diffusion Models" - CVPR Workshop on Data-Driven Autonomous Driving Simulation (DDADS)

<p align="center">
  <img src="assets/teaser_video.gif" alt="MObI Demo" width="80%"/>
</p>

## Abstract

Safety-critical applications, such as autonomous driving, require extensive multimodal data for rigorous testing. Methods based on synthetic data are gaining prominence due to the cost and complexity of gathering real-world data but require a high degree of realism and controllability in order to be useful. This paper introduces MObI, a novel framework for **M**ultimodal **Ob**ject **I**npainting that leverages a diffusion model to create realistic and controllable object inpaintings across perceptual modalities, demonstrated for both camera and lidar simultaneously. Using a single reference RGB image, MObI enables objects to be seamlessly inserted into existing multimodal scenes at a 3D location specified by a bounding box, while maintaining semantic consistency and multimodal coherence. Unlike traditional inpainting methods that rely solely on edit masks, our 3D bounding box conditioning gives objects accurate spatial positioning and realistic scaling. As a result, our approach can be used to insert novel objects flexibly into multimodal scenes, providing significant advantages for testing perception models.

> **Note:** 🚧 This repository is currently under construction. Proper documentation, installation instructions, and usage examples will be provided soon. 🚧

## Features

- Joint inpainting across multiple modalities (RGB camera, lidar depth and intensity)
- Object insertion using just a single reference image
- 3D bounding box conditioning for accurate spatial positioning
- Improved controllability compared to traditional inpainting methods



## Architecture

<p align="center">
  <img src="assets/architecture.jpg" alt="MObI Architecture" width="100%"/>
</p>

MObI extends Paint-by-Example, a reference-based image inpainting method, to include bounding box conditioning and jointly generate camera and lidar perception inputs.

## Motivation

MObI addresses limitations in existing approaches:

1. **Object inpainting methods based on edit masks alone** (e.g., Paint-by-Example) achieve high realism but can lead to surprising results because there are often multiple semantically consistent ways to inpaint an object within a scene.

2. **Methods based on 3D reconstruction** (e.g., NeuRAD) have strong controllability but sometimes lead to low realism, especially for object viewpoints that have not been observed.

## Installation

```bash
git clone https://github.com/alexbuburuzan/MObI.git
cd MObI
conda env create -f environment.yml
```

## Usage

```python
# Example scripts for using MObI will be provided here
```

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{buburuzan2025mobi,
  title={MObI: Multimodal Object Inpainting Using Diffusion Models},
  author={Buburuzan, Alexandru and Sharma, Anuj and Redford, John and Dokania, Puneet K and Mueller, Romain},
  journal={arXiv preprint arXiv:2501.03173},
  year={2025}
}
```

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Work done during Alexandru Buburuzan's internship at FiveAI
