## Reproduce PokeBNN on a Cloud TPU VM

### Step 1: Download ImageNet Dataset

Please refer to [this documentation](https://github.com/ychzhang/aqt/tree/main/aqt/jax_legacy/jax/imagenet#reproduce-aqt-experiments-on-gcloud-tpus) for setting up the gcloud command line tool, the cloud storage bucket, and downloading ImageNet.

### Step 2: Create TPU VM and Train PokeBNN-1.0x

The provided bash script `cloudtpu_train_pokebnn.sh` automates this step. Please copy the bash script to your local machine with gcloud command line tool installed and run the following command. Note that this script is "stand alone". It only requires gcloud CLI tool.
```
source cloudtpu_train_pokebnn.sh
```


### Visualizing Tensorboard Data through Colab

Run the following code block in Colab to visualize the tensorboard data. Fill in the cloud storage bucket name and directory name that stores the data.
```
from google.colab import auth
auth.authenticate_user()

#@markdown Enter cloud storage bucket name:
bucket_name = 'your_cloud_storage_bucket_name' #@param {type:"string"}
# list file in the bucket as a test
!gsutil ls gs://{bucket_name}/
#@markdown Enter log directory name in the cloud storage bucket:
log_dir = 'directory_in_bucket_storing_TB_data' #@param {type:"string"}

# copy tensorboard data to the temporary storage on colab
!mkdir /content/tb_dir
!gsutil rsync -r gs://{bucket_name}/{log_dir} /content/tb_dir

# load tensorboard
%load_ext tensorboard
%tensorboard --logdir /content/tb_dir
```


