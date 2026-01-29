# ALCHIMIA
This repository contains the code of the method described in "A Reinforcement Learning-guided Genetic Algorithm Integrating Medicinal Chemistry-inspired Molecular Transformations" (submitted).

## Requisites
This package requires:
* torch
* pandas
* tqdm
* rdkit

## Usage
### Training
python train.py --mode train --fixed_training_set --data train.csv --out_dir runs/t1 --epochs 5 --episodes_per_src 15 --max_steps 3 --temperature 1.0 --top_p 0.95 --w_sa 0.5 --w_qed 0.5 --fp_kind morgan --fp_nbits 2048 --fp_radius 2 --src_per_epoch 100 --device cuda:1

python train.py --mode train --data start.csv --out_dir runs/t2 --epochs 5 --episodes_per_src 15 --max_steps 3 --temperature 1.0 --top_p 0.95 --w_sa 0.5 --w_qed 0.5 --fp_kind avalon --fp_nbits 2048 --fp_radius 2 --src_per_epoch 100 --device cuda:1 --resume_ckpt runs/t1/policy_epoch_5.pt


### Generation
python train.py --mode sample --out_prefix runs/t3 --gen_max_steps 3 --gen_temperature 1.0 --gen_top_p 0.95 --n_samples 20 --src "CC(C)Cc1ccc(cc1)C(C)C(=O)O" --fp_kind avalon --fp_nbits 2048 --fp_radius 2 --device cuda:1 --ckpt runs/t2/policy_epoch_5.pt

