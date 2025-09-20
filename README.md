<div align="center">

  <h1><img src="resources/images/icon.png" width="35"> ganviz </h1>

![GitHub top language](https://img.shields.io/github/languages/top/Florian-barthel/splatviz) ![GitHub Release](https://img.shields.io/github/v/release/Florian-Barthel/splatviz) ![GitHub last commit](https://img.shields.io/github/last-commit/Florian-Barthel/splatviz) ![Static Badge](https://img.shields.io/badge/Platform-Linux-green) ![Static Badge](https://img.shields.io/badge/Platform-Windows-green)

</div>

![](resources/images/teaser.gif)

## Install

### 1. Download

Clone repository **recursively** in order to install glm from the diff_gaussian_rasterization package.

```bash
git clone https://github.com/Florian-Barthel/ganviz.git --recursive
```


### 2. Install

Create environment with <b>conda</b>:

```bash
conda env create -f environment.yml
conda activate gs-view

git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

Alternatively, use <b>micromamba</b>:

```bash
micromamba env create --file environment.yml --channel-priority flexible -y
micromamba activate gs-view

git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

Finally run the install script for the GAN preprocessor:
```bash
cd gan_preprocessing
./build.sh
```

## Launch

First check out the cgs-gan repository: <a href="https://github.com/fraunhoferhhi/cgs-gan">https://github.com/fraunhoferhhi/cgs-gan</a> and download the network checkpoints: [ffhq_512.pkl](https://huggingface.co/anonym892312603527/neurips25/resolve/main/models/ffhq_512.pkl?download=true), [ffhq_1024.pkl](https://huggingface.co/anonym892312603527/neurips25/resolve/main/models/ffhqc_1024.pkl?download=true) or [ffhq_2048.pkl](https://huggingface.co/anonym892312603527/neurips25/resolve/main/models/ffhqc_2048.pkl?download=true). Then run the main file and specify the path to the cgs-gan repository:
```bash
python run_main.py --gan_path=path/to/cgs-gan
```

In some cases you will have to add this variable so that opengl uses the correct version:
`export MESA_GL_VERSION_OVERRIDE=3.3`

## Widgets


### 🧭 Latent Widget
Simply drag the white dot across the 2D plane to interpolate in the latent space of the GAN.
<br>
<img src="resources/images/gan_mode.png" style="width: 600px;">

### Inversion Widget
Invert your own 3D head like in this [video](https://www.linkedin.com/posts/florian-barthel-9583b9208_we-have-just-released-a-new-feature-for-activity-7345716193535614979--nU8?utm_source=share&utm_medium=member_desktop&rcm=ACoAADS9oRQBL7WTKc4KVRY4d66D9oR51YDpUqc)

## ⭐ Recent Features

**_Version 1.2.0_**

- 2025-05-26: Added GAN mode.

**_Version 1.1.0_**

- 2024-08-12: Added a new Training Widget to inspect live training stats and to pause training
- 2024-08-11: Attach to a running 3DGS training
- 2024-08-10: Refactor rendering class for easier addition of new renderer
- 2024-08-07: Better Mouse Control (translate with middle mouse button)
- 2024-08-05: Allow editing of existing sliders
- 2024-07-30: Store slider values in presets
- 2024-07-28: New Performance Widget
- 2024-07-28: Editor now highlights special variables (gs, self, slider) and gives tooltips

**_Version 1.0.0_**

- 2024-07-12: Rebuild the whole application with imgui_bundle
- 2024-07-05: Compare two or more Gaussian Splatting scenes side by side

## Contribute

You are more than welcome to add further functionality or a better design to this interactive viewer!
The main goal is to create an easy-to-use tool that can be applied for debugging and for understanding
3D Gaussian Splatting objects.
For reformating code please use [black](https://github.com/psf/black) with --line-length 120.

## Citation

If you find this viewer useful, please consider citing our work:

```
@misc{barthel2024gaussian,
    title={Gaussian Splatting Decoder for 3D-aware Generative Adversarial Networks}, 
    author={Florian Barthel and Arian Beckmann and Wieland Morgenstern and Anna Hilsmann and Peter Eisert},
    year={2024},
    eprint={2404.10625},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## References

This viewer is inspired by the visualizer from Efficient Geometry-aware 3D Generative Adversarial
Networks (EG3D).

- GUI: <a href="https://pyimgui.readthedocs.io/en/latest/guide/first-steps.html">pyimgui</a> and
<a href="https://github.com/pthom/imgui_bundle">imgui_bundle</a> which are python wrappers for the c++ library
<a href="https://github.com/ocornut/imgui">ImGUI</a>
- Original code base: <a href="https://github.com/NVlabs/eg3d">EG3D</a>
- 3DGS: <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"> 3D Gaussian Splatting</a>
- Compressing 3DGS scenes: <a href="https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/">Compact 3D Scene
Representation via Self-Organizing Gaussian Grids</a>
- 3DGS Rasterizer with depth and alpha: <a href="https://github.com/slothfulxtx/diff-gaussian-rasterization">Diff
rasterizer with depth and alpha</a>
