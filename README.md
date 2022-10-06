# Diffusion Models for MNIST Data

This is an adjusted version of the diffusion models implementation for use with MNSIT handwritten digit data. The goal here is to train a model to learn how to draw handwritten digits. `test.ipynb` can be used to adjust and train the model and can be run e.g. on [Google Colab by clicking here](https://colab.research.google.com/github/jonasloos/Diffusion-Models-pytorch/blob/main/test.ipynb). Just uncomment the cells relevant for Colab.

### Examples

Epoch | Example Images
-|-
0 | ![0](https://user-images.githubusercontent.com/33965649/194434345-c76d4411-5bfe-41e3-9eb7-4be04f93f6af.jpg)
20 | ![20](https://user-images.githubusercontent.com/33965649/194434377-4a914083-3346-46c3-9761-9d00dc8918de.jpg)
50 | ![50](https://user-images.githubusercontent.com/33965649/194434565-b59a5ff8-5da2-43e9-936e-3570a5213821.jpg)
100 | ![100](https://user-images.githubusercontent.com/33965649/194434580-b9a8977c-4935-49fe-ab4b-cf12b974696c.jpg)
150 | ![150](https://user-images.githubusercontent.com/33965649/194434678-95617034-e63d-4162-87f1-21da3e7d60d5.jpg)
200 | ![200](https://user-images.githubusercontent.com/33965649/194434595-ff57edc2-af38-4fef-b28a-48aecbb914dc.jpg)
250 | ![250](https://user-images.githubusercontent.com/33965649/194434654-3e29c6e1-c351-4cfb-889a-fc5cce48513e.jpg)


# Original Readme

This is an easy-to-understand implementation of diffusion models within 100 lines of code. Different from other implementations, this code doesn't use the lower-bound formulation for sampling and strictly follows Algorithm 1 from the [DDPM](https://arxiv.org/pdf/2006.11239.pdf) paper, which makes it extremely short and easy to follow. There are two implementations: `conditional` and `unconditional`. Furthermore, the conditional code also implements Classifier-Free-Guidance (CFG) and Exponential-Moving-Average (EMA). Below you can find two explanation videos for the theory behind diffusion models and the implementation.

<a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">
   <img alt="Qries" src="https://user-images.githubusercontent.com/61938694/191407922-f613759e-4bea-4ac9-9135-d053a6312421.jpg"
   width="300">
</a>

<a href="https://www.youtube.com/watch?v=TBCRlnwJtZU">
   <img alt="Qries" src="https://user-images.githubusercontent.com/61938694/191407849-6d0376c7-05b2-43cd-a75c-1280b0e33af1.png"
   width="300">
</a>

<hr>

## Train a Diffusion Model on your own data:
### Unconditional Training
1. (optional) Configure Hyperparameters in ```ddpm.py```
2. Set path to dataset in ```ddpm.py```
3. ```python ddpm.py```

### Conditional Training
1. (optional) Configure Hyperparameters in ```ddpm_conditional.py```
2. Set path to dataset in ```ddpm_conditional.py```
3. ```python ddpm_conditional.py```

## Sampling
The following examples show how to sample images using the models trained in the video. You can download the checkpoints for the models [here](https://drive.google.com/drive/folders/1beUSI-edO98i6J9pDR67BKGCfkzUL5DX?usp=sharing).
### Unconditional Model
```python
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("unconditional_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    plot_images(x)
```

### Conditional Model
This model was trained on CIFAR-10 64x64 with 10 classes ```airplane:0, auto:1, bird:2, cat:3, deer:4, dog:5, frog:6, horse:7, ship:8, truck:9```
```python
    n = 10
    device = "cuda"
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load("conditional_ema_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    y = torch.Tensor([6] * n).long().to(device)
    x = diffusion.sample(model, n, y, cfg_scale=3)
    plot_images(x)
```
