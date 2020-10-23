# MHSAN: Multi-Head Self-Attention Network for Visual Semantic Embedding
This repository is the official PyTorch implementation of MHSAN: Multi-Head Self-Attention Network for Visual Semantic Embedding (WACV2020) 
by [Geondo Park](https://github.com/GeondoPark), Chihye Han, Wonjun Yoon, Daeshik Kim.

## Prepare Dataset
Download the dataset files: [coco](https://cocodataset.org/#download), [Flickr](http://shannon.cs.illinois.edu/DenotationGraph/)

We use splits produced by [Andrej Karpathy](https://cs.stanford.edu/people/karpathy/deepimagesent/). 
Download the json file from [here](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
Then, put it same directory, respectively. (ref, get_data_path in utils.py)

## Train Coco
For train the new model with coco dataset, run train_coco.py
Overall hyperparameters for training are set by default.
```
python train_coco.py -hop 10 -name <model_name> -p-coeff 0.1
```
## Train flickr
For train the new model with flickr dataset, run train_flickr.py
Overall hyperparameters for training are set by default.
```
python train_flickr.py -hop 10 -name <model_name> -p-coeff 0.1
```

## Implementation
 - Our code is implemented based on : [vsepp](https://github.com/fartashf/vsepp)

## Citation
```
@inproceedings{park2020mhsan,
  title={MHSAN: Multi-Head Self-Attention Network for Visual Semantic Embedding},
  author={Park, Geondo and Han, Chihye and Yoon, Wonjun and Kim, Daeshik},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1518--1526},
  year={2020}
}
```
