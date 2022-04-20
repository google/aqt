## Reproduce PokeBNN on a Cloud TPU VM

This readme describes how to reproduce the [PokeBNN paper](https://arxiv.org/abs/2112.00133) on a google cloud TPU virtual machine.

### Step 1: Download ImageNet Dataset

Please refer to the section `Reproduce AQT Experiments on GCloud TPUs` in [this documentation](https://github.com/ychzhang/aqt/tree/main/aqt/jax_legacy/jax/imagenet#reproduce-aqt-experiments-on-gcloud-tpus) for setting up the gcloud command line tool, the cloud storage bucket, and downloading the ImageNet dataset.

### Step 2: Create TPU VM and Train PokeBNN-1.0x

The provided bash script `cloudtpu_train_pokebnn.sh` automates this step. Please copy the bash script to your local machine with gcloud command line tool installed and run the following command. Note that this script is "stand alone". It only requires gcloud CLI tool.
```
source cloudtpu_train_pokebnn.sh
```

### Step 3: Visualize Tensorboard Data through Colab

1. Navigate to https://colab.research.google.com/
2. Open a new colab file. Load and run the provided `tensorboard.ipynb`. Please remember filling in the correct cloud storage bucket name and directory name that stores the data.

## Citation

If you find PokeBNN and this repository useful, please cite:

```
@article{zhanglew2021pokebnn,
  title={PokeBNN: A Binary Pursuit of Lightweight Accuracy},
  author={Zhang, Yichi and Zhang, Zhiru and Lew, Lukasz},
  journal={arXiv preprint arXiv:2112.00133},
  year={2021}
}
```

