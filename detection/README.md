# Installation
We use the [*MMYOLO*](https://github.com/open-mmlab/mmyolo) platform to train on our dataset. It features several single-stage detectors that can be trained on custom datasets using the COCO format. 

Detailed installation process for MMYOLO is given in this [link](https://github.com/open-mmlab/mmyolo/blob/main/docs/en/get_started/installation.md). 


```shell
conda create -n mmyolo python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmyolo
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc6,<3.1.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```

We have trained using 2 models: [YoloV8](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov8) and [YoloX](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox). Both configuration files are located in this folder. To train the models using the configs, first move the configuration file to the corresponding folder in the `configs` folder in `mmyolo`.

For example after moving the `yolox_skin.py` file to the `yolox` folder, relative path of the config file becomes as follows: `mmyolo/configs/yolox/yolox_skin.py`. 

Finally to launch the training process with 4 gpus, run the following from the main folder of `mmyolo`
```shell
./tools/dist_train.sh configs/yolox/yolox_skin.py 4
```

Weights will be saved in the `workdirs` folder under the name of configuration file.