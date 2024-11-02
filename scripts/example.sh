# !/usr/bin/env bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mamba
cd /home/zkl/Documents/Codes/MAGR-github

# run the following command in the terminal
TIRCH_DISTRIBUTED_DEBUG=DETAIL torchrun \
	--nproc_per_node 2 --master_port 29503 main.py \
	--config ./configs/mtl.yaml \
	--model fea_gr --dataset class-mtl \
	--batch_size 5 --minibatch_size 3 \
	--n_tasks 5 --n_epochs 50 --gpus 2 3 \
	--base_pretrain True --fewshot True \
	--buffer_size 50 
