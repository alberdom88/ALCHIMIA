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

`python train.py --mode train --data <training_set_file> --out_dir <output_dir> --epochs <epochs> --episodes_per_src <episodes> --max_steps <max_steps> --lr <lr> --temperature <temperature> --top_p <top_p> --w_sa <w_sa> --w_qed <w_qed> --fp_kind <fp_kind> --fp_nbits <fp_nbits> --fp_radius <fp_radius> --src_per_epoch <src_per_epoch> --device <device>` --resume_ckpt <resume_ckpt>`

`<training_set_file>`: training set file (for curriculum learning use start.csv and add the flag --fixed_training_set)\
`<output_dir>`: output directory\
`<epochs>`: number of training epochs\
`<episodes>`: number of smiles generated for single molecule\ 
`<max_steps>`: maximum number of applied molecular trasformations (M)\
`<lr>`: learning rate\
`<temperature>`: temperature scaling\
`<top_p>`: top-p (nucleus) sampling\
`<w_sa>`: reward weight for SA score\
`<w_qed>`: reward weight for QED score\
`<fp_kind>`: molecular fingerprint type ("morgan","feat_morgan","rdk","pattern","layered","maccs","atompair","torsion" or "avalon")\
`<fp_nbits>`: number of bits of the molecular fingerprint\
`<fp_radius>`: fingerprint radius (for "morgan" and "feat_morgan")\
`<src_per_epoch>`: number of molecules considered for each training epoch\
`<device>`: "cpu" or "cuda"\
`<resume_ckpt>`: pretrained model path to continue the training from a checkpoint

Examples:

`python train.py --mode train --fixed_training_set --data train.csv --out_dir run/ --epochs 10 --episodes_per_src 15 --max_steps 3 --temperature 1.0 --top_p 0.95 --w_sa 0.5 --w_qed 0.5 --fp_kind morgan --fp_nbits 2048 --fp_radius 2 --src_per_epoch 100 --device cuda:1

`python train.py --mode train --data start.csv --out_dir runs/t2 --epochs 10 --episodes_per_src 15 --max_steps 3 --temperature 1.0 --top_p 0.95 --w_sa 0.5 --w_qed 0.5 --fp_kind avalon --fp_nbits 2048 --src_per_epoch 100 --device cuda:1 --resume_ckpt models/policy_m3.pt`

### Generation
`python train.py --mode sample --out_prefix runs/t3 --gen_max_steps 3 --gen_temperature 1.0 --gen_top_p 0.95 --n_samples 20 --src "CC(C)Cc1ccc(cc1)C(C)C(=O)O" --fp_kind avalon --fp_nbits 2048 --fp_radius 2 --device cuda:1 --ckpt runs/t2/policy_epoch_5.pt`


