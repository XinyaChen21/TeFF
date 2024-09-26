<div align="center">

<h1>TeFF: Learning 3D-Aware GANs from Unposed Images with Template Feature Field (ECCV 2024 Oral)</h1>

<div>
Xinya Chen<sup>1</sup>, Hanlei Guo<sup>1</sup>, Yanrui Bin<sup>2</sup>, Shangzhan Zhang<sup>1</sup>, Yuanbo Yang<sup>1</sup>, Yue Wang<sup>1</sup>, Yujun Shen<sup>3</sup>, Yiyi Liao<sup>1*</sup>
</div>
<div>
    <sup>1</sup>Zhejiang University&emsp; <sup>2</sup>The Hong Kong Polytechnic University&emsp; <sup>3</sup>Ant Group&emsp; <sup>*</sup>corresponding author
</div>

<h4 align="center">
  <a href="https://xdimlab.github.io/TeFF/" target='_blank'>[Project Page]</a>
  <a href="https://arxiv.org/abs/2404.05705" target='_blank'>[paper]</a>
</h4>

<img src="./assets/teaser_final.png" width="500">

**Figure:** Overview of TeFF.

</div>


## Requirements
We test our code on PyTorch 1.12.1 and CUDA toolkit 11.3. Please follow the instructions on [https://pytorch.org](https://pytorch.org) for installation. Other dependencies can be installed with the following command:
```Shell
    conda env create -f environment.yml
    conda activate TeFF
```


## Training
We provide the scripts for training in ./shell_scripts. Please update the path to the dataset in the scripts. 

## Inference
The pretrained models are available [here](https://drive.google.com/drive/folders/1_vX1QZtDgP21mGikc8UZHvGjkXMPhACL?usp=sharing).

To perform the inference, you can use the command:
```Shell
    python gen_videos.py --outdir=${OUTDIR}
            --trunc=0.7 
            --seeds=0-8 
            --grid=3x3 
            --network=${CHECKPOINT_PATH} 
            --reload_modules=True
```
where `CHECKPOINT_PATH` should be replaced with the path to the checkpoint. `OUTDIR` denotes the folder to save the outputs.

## BibTeX

```bibtex
@article{Chen2024TeFF,
    author    = {Chen, Xinya and Guo, Hanlei and Bin, Yanrui and Zhang, Shangzhan and Yang, Yuanbo and Wang, Yue and Shen, Yujun and Liao, Yiyi},
    title     = {Learning 3D-Aware GANs from Unposed Images with Template Feature Field},
    journal = {arXiv preprint arXiv:2404.05705},
    year = {2024}}
```
