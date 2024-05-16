#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu
#BSUB -n 64 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/eb_count_%J.out # output file
#BSUB -e logs/eb_count_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_count # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo '--- Start computing model'
# python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/train.py \
python3 $cwd/train.py \
--train_mode count \
--split False \
--splitting_mode random \
--output_dir './T_perturb/T_perturb/plt/res/eb' \
--ckpt_masking_path './T_perturb/T_perturb/Model/checkpoints/20240515_1906_petra'\
'_train_masking_lr_0.001_wd_0.0001_batch_32_'\
'mlmp_0.15_tp_1-2-4.ckpt' \
--src_dataset './T_perturb/T_perturb/pp/res/eb/dataset_hvg_src/Day 00-03.dataset' \
--tgt_dataset_folder './T_perturb/T_perturb/pp/res/eb/dataset_hvg_tgt' \
--src_adata './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad' \
--tgt_adata_folder './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt' \
--mapping_dict_path  './T_perturb/T_perturb/pp/res/eb/token_id_to_genename_hvg.pkl' \
--batch_size 32 \
--max_len 263 \
--epochs 100 \
--tgt_vocab_size 2001 \
--petra_lr 0.0001 \
--count_lr 0.00005 \
--petra_wd 0.0001 \
--count_wd 0.01 \
--mlm_prob 0.15 \
--n_workers 64 \
--num_layers 1 \
--d_ff 16 \
--loss_mode zinb \
--time_steps 1 2 4 \
--var_list Time_point
echo '--- Finished computing model'
