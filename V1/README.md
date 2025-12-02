

# quick start

自建环境使用conda或uv都行

```
pip install -r requirements.txt
或
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install dependencies:  安装依赖项：
uv sync
# or
uv pip install -r pyproject.toml
```

代码运行参考下例，使用`uv`就用`uv run train_cifar10.py`

`ResNet_18.ipynb` 和`ResNet_18.py` 为小组同学贡献的可以快速了解resnet18，代码可直接运行

模型训练建议使用GPU和linux系统,或者wsl

# Usage example

`pip install -r requirements.txt` # install dependencies

`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py --dataset cifar100` # cifar-100

`python train_cifar10.py  --size 48` # vit-patchsize-4-imsize-48

`python train_cifar10.py --patch 2` # vit-patchsize-2

`python train_cifar10.py --net vit_small --n_epochs 400` # vit-small

`python train_cifar10.py --net vit_timm` # train with pretrained vit

`python train_cifar10.py --net dyt` # train with Layernorm-less ViT (DyT)

`python train_cifar10.py --net convmixer --n_epochs 400` # train with convmixer

`python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3`

`python train_cifar10.py --net cait --n_epochs 200` # train with cait

`python train_cifar10.py --net swin --n_epochs 400` # train with SwinTransformers

`python train_cifar10.py --net res18` # resnet18+randaug

# Results

| CIFAR10 | Accuracy | Train Log |
|:-----------:|:--------:|:--------:|
| ViT patch=2 |    80%    | |
| ViT patch=4 Epoch@200 |    80%   | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r) |
| ViT patch=4 Epoch@500 |    85%   | [Log](https://api.wandb.ai/links/arutema47/wrfsfmlo) |
| ViT patch=4 Epoch@1000 |    89%   | [Log](https://api.wandb.ai/links/arutema47/sr9eph7v) |
| ViT patch=8 |    30%   | |
| ViT small  | 80% | |
| DyT |    74%   | [Log](https://api.wandb.ai/links/arutema47/9lsyl4u0) |
| MLP mixer |    88%   | |
| CaiT  | 80% | |
| Swin-t  | 90% | |
| ViT small (timm transfer) | 97.5% | |
| ViT base (timm transfer) | 98.5% | |
| [ConvMixerTiny(no pretrain)](https://openreview.net/forum?id=TVHS5Y4dNvM) | 96.3% |[Log](https://wandb.ai/arutema47/cifar10-challange/reports/convmixer--VmlldzoyMjEyOTk1?accessToken=2w9nox10so11ixf7t0imdhxq1rf1ftgzyax4r9h896iekm2byfifz3b7hkv3klrt)|
|   resnet18  |  93%  | |
|   resnet18+randaug  |  95%  | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTYz?accessToken=968duvoqt6xq7ep75ob0yppkzbxd0q03gxy2apytryv04a84xvj8ysdfvdaakij2) |

| CIFAR100 | Accuracy | Train Log |
|:-----------:|:--------:|:--------:|
| ViT patch=4 Epoch@200 |    52%   | [Log](https://api.wandb.ai/links/arutema47/f8mz3mpk) |
| resnet18+randaug |    71%   | [Log](https://wandb.ai/arutema47/cifar-challenge/reports/Res18-CIFAR100--VmlldzoxMjUwNzU3Mg?accessToken=fw9ojmpfuqrrxjers2duixssezqifaonvbmf8x3ynieldw3auh53ax992g0z6cx3) |

# Model Export

This repository supports exporting trained models to ONNX and TorchScript formats for deployment purposes. You can export your trained models using the `export_models.py` script.

### Basic Usage

```bash
python export_models.py --checkpoint path/to/checkpoint --model_type vit --output_dir exported_models

### citation
`@misc{yoshioka2024visiontransformers,
  author       = {Kentaro Yoshioka},
  title        = {vision-transformers-cifar10: Training Vision Transformers (ViT) and related models on CIFAR-10},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/kentaroy47/vision-transformers-cifar10}}
}`
