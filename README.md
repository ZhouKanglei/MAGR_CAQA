<div align="center">
  <div>
    <h1>
        MAGR: Manifold-Aligned Graph Regularization for Continual Action Quality Assessment
    </h1>
  </div>
  <div>
      Kanglei Zhou &emsp; 
      <a href='https://lywang3081.github.io/'>Liyuan Wang</a>  &emsp; 
      <a href='https://indussky8.github.io/'>Xingxing Zhang</a> &emsp; 
      <a href='http://hubertshum.com/'>Hubert P. H. Shum</a> <br/>
      <a href='https://frederickli.webspace.durham.ac.uk/'>Frederick W. B. Li &emsp; 
      <a href='https://baike.baidu.com/item/%E6%9D%8E%E5%BB%BA%E5%9B%BD/62860598?fr=ge_ala'>Jianguo Li</a>  &emsp; 
      <a href='https://orcid.org/0000-0001-6351-2538'>Xiaohui Liang</a>
  </div>
  <br/>
  <div>
  ECCV 2024 Oral Presentation > 
  <a href='./supp/MAGR-poster.pdf'> Poster </a> |
  <a href='./supp/MAGR-slides.pdf'> Slides </a>
  </div>
  <br/>
</div>

This repository contains the implementation of MAGR, a novel approach designed for Continual Action Quality Assessment (CAQA). MAGR leverages Manifold Projector (MP) and Intra-Inter-Joint Graph Regularization (IIJ-GR) to address the challenges of feature deviation and regressor confusion across incremental sessions. The method aims to adapt to real-world complexities while safeguarding user privacy.

![](framework.png)




## Requirement

- torch==1.12.0 
- torchvision==0.13.0
- torchvideotransforms 
- tqdm  
- numpy  
- scipy  
- quadprog

## Usage

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Preparing datasets:

To get started with the experiments, follow the steps below to prepare the datasets:

#### MTL-AQA
1. Download the MTL-AQA dataset from the [MTL-AQA repository](https://github.com/ParitoshParmar/MTL-AQA).
2. Organize the dataset in the following structure:

```
$DATASET_ROOT
├── MTL-AQA/
    ├── new
        ├── 01
        ...
        └── 26
    ├── info
        ├── final_annotations_dict_with_dive_number
        ├── test_split_0.pkl
        └── train_split_0.pkl
    └── model_rgb.pth
```

#### UNLV-Dive

1. Download the AQA-7 dataset:

```
mkdir AQA-Seven & cd AQA-Seven
wget http://rtis.oit.unlv.edu/datasets/AQA-7.zip
unzip AQA-7.zip
```

2. Organize the dataset as follows:

```
$DATASET_ROOT
├── Seven/
    ├── diving-out
        ├── 001
            ├── img_00001.jpg
            ...
        ...
        └── 370
    ├── gym_vault-out
        ├── 001
            ├── img_00001.jpg
            ...
    ...

    └── Split_4
        ├── split_4_test_list.mat
        └── split_4_train_list.mat
```
#### JDM-MSA

Contact the corresponding author of [the JDM-MSA paper](https://ieeexplore.ieee.org/document/10049714) to obtain access to the dataset. You may need to complete a form before using this dataset for academic research.


### Pre-trained model:

Please download [the pre-trained I3D model](https://github.com/hassony2/kinetics_i3d_pytorch/tree/master/model) and then put it to `weights/model_rgb.pth`.

### Training from scratch:

We provide for training our model in both distributed and dataparallel modes. Here's a breakdown of each command:

1. In distributed mode:

```bash
torchrun \
	--nproc_per_node 2 --master_port 29505 main.py \
	--config ./configs/{config file name}.yaml \
	--model {model name: joint/adam/fea_gr} \
	--dataset {dataset name: class-mtl/class-aqa/class-jdm} \
	--batch_size 5 --minibatch_size 3 --n_tasks 5 --n_epochs 50 \
    --fewshot True --buffer_size 50 \
    --gpus 0 1
```

2. In dataparallel mode

```bash
python main.py \
	--config ./configs/{config file name}.yaml \
	--model {model name: joint/adam/fea_gr} \
	--dataset {dataset name: class-mtl/class-aqa/class-jdm} \
	--batch_size 5 --minibatch_size 3 --n_tasks 5 --n_epochs 50 \
    --fewshot True --buffer_size 50 \
    --gpus 0 1
```

Choose the appropriate command based on your training setup (distributed or dataparallel) and adjust the configurations as needed.

### Evaluation:

If you want to perform evaluation using the same configurations as training but with the addition of the `--phase test` option.

A pretrained example is provided in this repository. You can find the pretrained logs and weights in the directory `outputs/zkl-fscl/class-mtl/fea_gr05-buffer50`. This model is trained using the base session pretrained weights located at `weights/class-mtl05.pth`. In practice, training a robust base session model allows us to build on these weights for further experiments, saving on additional training time.

To evaluate the pretrained model, use the following command:

```bash
torchrun \
	--nproc_per_node 2 --master_port 29503 main.py \
	--config ./configs/mtl.yaml \
	--model fea_gr --dataset class-mtl \
	--batch_size 5 --minibatch_size 3 \
	--n_tasks 5 --n_epochs 50 --gpus 2 3 \
	--base_pretrain True --fewshot True \
	--buffer_size 50 --phase test \
    --exp_name fea_gr05-buffer50
```

Logs for this evaluation can be found in the `outputs/zkl-fscl/class-mtl/fea_gr05-buffer50/logs` directory.

## Acknowledgements

This repository is based on [mammoth](https://github.com/aimagelab/mammoth), many thanks.

If you have any specific questions or if there's anything else you'd like assistance with regarding the code, feel free to let us know. 