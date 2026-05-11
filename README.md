<div align="center">

  <h1><img src="resources/images/icon.png" width="35"> ganviz </h1>

![GitHub top language](https://img.shields.io/github/languages/top/Florian-barthel/splatviz) ![GitHub Release](https://img.shields.io/github/v/release/Florian-Barthel/splatviz) ![GitHub last commit](https://img.shields.io/github/last-commit/Florian-Barthel/splatviz) ![Static Badge](https://img.shields.io/badge/Platform-Linux-green) ![Static Badge](https://img.shields.io/badge/Platform-Windows-green)

</div>

![](resources/images/teaser.gif)

## Install

## Install

Install `uv` first if it is not already available. See the official
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/), or use the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the CUDA Toolkit:
[11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
[12.1](https://developer.nvidia.com/cuda-12-1-1-download-archive)
[12.4](https://developer.nvidia.com/cuda-12-4-1-download-archive)
[12.6](https://developer.nvidia.com/cuda-12-6-3-download-archive)
[12.8](https://developer.nvidia.com/cuda-12-8-2-download-archive)
[12.9](https://developer.nvidia.com/cuda-12-9-1-download-archive)
[13.0](https://developer.nvidia.com/cuda-13-0-3-download-archive)

After installing `uv` and CUDA, clone and install splatviz:

```bash
git clone https://github.com/Florian-Barthel/splatviz.git
cd splatviz

uv sync --group <cuda-group>

uv pip install --no-build-isolation git+https://github.com/ashawkey/diff-gaussian-rasterization.git
```

Replace `<cuda-group>` with the `uv` CUDA group that matches your system:

```bash
uv sync --extra cu118
uv sync --extra cu126
uv sync --extra cu128
uv sync --extra cu130
```

Clone CGS-GAN:
```bash
git clone https://github.com/fraunhoferhhi/cgs_gan.git --single-branch
```
and download a pretrained checkpoint:
[ffhq_512.pkl](https://huggingface.co/Fubei/cgs_gan_checkpoints/resolve/main/ffhqc_512.pkl?download=true)

## Launch

```bash
python run_main.py
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

- 2025-09-21: Inversion Tool
- 2025-09-21: Latent Space widget

## Contribute

You are more than welcome to add further functionality or a better design to this interactive viewer!
The main goal is to create an easy-to-use tool that can be applied for debugging and for understanding
3D Gaussian Splatting objects.
For reformating code please use [black](https://github.com/psf/black) with --line-length 120.

## Citation

If you find this viewer useful, please consider citing our work:

```
@misc{barthel2025cgsgan,
      title={CGS-GAN: 3D Consistent Gaussian Splatting GANs for High Resolution Human Head Synthesis}, 
      author={Florian Barthel and Wieland Morgenstern and Paul Hinzer and Anna Hilsmann and Peter Eisert},
      year={2025},
      eprint={2505.17590},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.17590}, 
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
