# TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning [[website]](https://hanlab.mit.edu/projects/tinyml/tinyTL/)

## On-Device Learning, not Just Inference
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/on-device-learning.png" width="80%" />
</p>

## Activation is the Main Bottleneck, not Parameters
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/acitvation-is-the-bottleneck.png" width="70%" />
</p>

## Tiny Transfer Learning
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/tinyTL.png" width="70%" />
</p>

## Transfer Learning Results
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/tinytl_results_8.png" width="80%" />
</p>

## Combining with Batch Size 1 Training
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/tinytl_batch1.png" width="80%" />
</p>

## Data Preparation
To set up the datasets, please run `bash make_all_datasets.sh` under the folder **dataset_setup_scripts**.

## Requirement
* Python 3.6+
* Pytorch 1.4.0+

## How to Run Transfer Learning Experiments
To run transfer learning experiments, please first set up the datasets and then run **tinytl_fgvc_train.py**. 
Scripts are available under the folder **exp_scripts**. 

## TODO

- [ ] Add system support for TinyTL 

```BibTex
@inproceedings{
  cai2020tinytl,
  title={TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning},
  author={Cai, Han and Gan, Chuang and Zhu, Ligeng and Han, Song},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
