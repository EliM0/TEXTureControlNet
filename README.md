# Adding ControlNet to TEXTure

In this repository we added [ControlNet](https://github.com/lllyasviel/ControlNet) to [TEXTure](https://github.com/TEXTurePaper/TEXTurePaper).

## Content

This repository is forked from the [TEXTure](https://github.com/TEXTurePaper/TEXTurePaper) repository so it has all the code present there. We also added the files from the [ControlNet](https://github.com/lllyasviel/ControlNet) repository to be able to use it. Then we added the files `controlnet_depth.py` and `prompt_config.yml`, and some changes around the TEXTure code to be able to use ControlNet in TEXTure.

## Installation

To install TEXTure with ControlNet first create a virtual environment with the requirements for ControlNet:

```bash
conda env create -f environment.yaml
conda activate texture
```

Then, install the requirements for TEXTure:

```bash
pip install -r requirements.txt
```

and Kaolin:

```bash
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/{TORCH_VER}_{CUDA_VER}.html
```

Note that you also need a token for StableDiffusion. 
First accept conditions for the model you want to use, default one is [`stabilityai/stable-diffusion-2-depth`]( https://huggingface.co/stabilityai/stable-diffusion-2-depth). Then, add a TOKEN file [access token](https://huggingface.co/settings/tokens) to the root folder of this project, or use the `huggingface-cli login` command.

To be able to run ControlNet, you will also need to add the [ControlNet depth model](https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_depth.pth), and put it inside the `src/models`.


## Running

We have an example that uses a 3D model avatar of Spider-Man generated with [ECON](https://github.com/YuliangXiu/ECON). To run it, move to the root folder of this repository and execute the following command:

```bash
python -m scripts.run_texture --config_path=configs/text_guided/spiderman_example.yaml
```

In the `experiments` folder you should find the results.

If you want to execute the basic TEXTure without ControlNet using this code you can add the `control_net: False` attribute to the `config file` you want to execute. For example:

```yaml
log:
  exp_name: spiderman_example
guide:
  text: "Amazing Spiderman, hyper realistic, {} view"
  append_direction: True
  shape_path: shapes/spiderman_example.obj
  control_net: False
optim:
  seed: 3
```
