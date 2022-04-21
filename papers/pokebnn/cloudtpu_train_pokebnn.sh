#!/bin/bash

# ------- Update the Variables -------

# Change VM_NAME to your cloud TPU VM name
VM_NAME=v3vm
# Change ZONE to your cloud TPU zone
ZONE=us-east1-d
# coud TPU VM type
ACC_TYPE=v3-128
# cloud TPU software version
SW_VERSION=tpu-vm-tf-2.8.0-pod
# Username on the cloud TPU VM
USERNAME=xxx
# Temporary directory on cloud TPU VM that
# stores model checkpoint and Tensorboard data
LOCAL_WORK_DIR=/home/$USERNAME/work_dir
# Directory in cloud storage bucket
# This with be synced with local_work_dir at the end of the training
GCS_WORK_DIR=gs://SAVE_DIR
# Path of ImageNet dataset in cloud storage bucket
GCS_TFDS_DATA_DIR=gs://ImageNet_DIR
# 8-bit ResNet50 checkpoint path
RESNET508B_CKPT=gs://aqt-resnet50-w8a8auto


#  ------- Optionally Change -------

# Training log file
TRAINING_LOG=/home/$USERNAME/training_log.txt
# Directory on cloud TPU that saves the training report
REPORT_DIR=$LOCAL_WORK_DIR/report
# Training configuration file
CONFIG_DICT=configs/pokebnn/pokebnn_config.py
# TMUX session name that will be launched on the cloud TPU VM
TMUX_SESSION=poke
# Command to be excuted after ssh into the cloud TPU VM
COMMAND="
        pip install 'jax[tpu]>=0.2.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
        pip install --upgrade clu &&
        pip install flax &&
        pip install dacite &&
        sudo apt-get install apt-transport-https ca-certificates gnupg &&
        echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - &&
        sudo apt-get update &&
        sudo apt-get install google-cloud-cli &&
        git clone https://github.com/google/aqt.git &&
        export TF_CPP_MIN_LOG_LEVEL=0 &&
        export PYTHONPATH=/home/$USERNAME/aqt &&
        export TFDS_DATA_DIR=$GCS_TFDS_DATA_DIR &&
        cd aqt/aqt/jax_legacy/jax/imagenet &&
        python3 train.py --model_dir $LOCAL_WORK_DIR --hparams_config_dict $CONFIG_DICT --report_dir $REPORT_DIR --batch_size 8192 --resnet508b_ckpt_path $RESNET508B_CKPT --config_idx 0 2>&1 | tee -a $TRAINING_LOG &&
        cp /home/$USERNAME/$TRAINING_LOG $LOCAL_WORK_DIR &&
        gsutil rsync -r -d $LOCAL_WORK_DIR $GCS_WORK_DIR
"

# create VM
gcloud alpha compute tpus tpu-vm create $VM_NAME --zone $ZONE --accelerator-type $ACC_TYPE --version $SW_VERSION

# execute "ls" on vm. This helps the local machine automatically update ssh key
gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all --command "ls"

# ssh and execute command on VM
gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all --command "tmux new-session -d -s $TMUX_SESSION '$COMMAND'"
