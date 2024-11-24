<p align="center">

  <h2 align="center"><strong>Dataset Distillers Are Good Label Denoisers In the Wild</strong></h2>

  <p align="center">
  <span>
    <a href="https://scholar.google.com/citations?user=PKFAv-cAAAAJ&hl=en">Lechao Cheng</a>,
    Kaifeng Chen,
    Jiyang Li,
    Shengeng Tang,
    Shufei Zhang,
    Meng Wang
  </span>
</p>


<div align="center">
<a href='https://arxiv.org/abs/2411.11924'><img src='https://img.shields.io/badge/arXiv-2411.11924-b31b1b.svg'></a>
</div>
</div>

## Getting Started

### Environment

You can create environment as follows

```
conda env create -f environment.yaml
conda activate ddlnl
```

### Dataset

For Tiny-ImageNet, it is best to download it in [tiny-imagenet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and then process it through `dataSolu/deal_Tiny.py`.

## Train

See `ddlnl_scripts.md` to reproduce the results

## Acknowledgement

Our code is built upon [DATM](https://github.com/NUS-HPC-AI-Lab/DATM), [DANCE](https://github.com/Hansong-Zhang/DANCE) and [RCIG](https://github.com/yolky/RCIG).

## Citation

If you find our code useful for your research, please cite our paper.

```
@article{cheng2024dataset,
  title={Dataset Distillers Are Good Label Denoisers In the Wild},
  author={Cheng, Lechao and Chen, Kaifeng and Li, Jiyang and Tang, Shengeng and Zhang, Shufei and Wang, Meng},
  journal={arXiv preprint arXiv:2411.11924},
  year={2024}
}
```

