#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/count_%J.out # output file
#BSUB -e logs/count_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_count # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo "--- Start computing model"

python3 $cwd/train.py \
--train_mode count \
--split False \
--generate False \
--splitting_mode stratified \
--output_dir "./T_perturb/T_perturb/plt/res/cytoimmgen" \
--ckpt_masking_path "./T_perturb/T_perturb/Model/checkpoints/"\
"20240508_1759_petra_train_masking_lr_0.001"\
"_wd_0.001_batch_64_mlmp_0.15_tp_1-3.ckpt/"\
"checkpoint/mp_rank_00_model_states.pt" \
--src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl" \
--batch_size 64 \
--max_len 300 \
--epochs 100 \
--tgt_vocab_size 1261 \
--petra_lr 0.001 \
--count_lr 0.0005 \
--petra_wd 0.001 \
--count_wd 0.001 \
--mlm_prob 0.15 \
--n_workers 32 \
--loss_mode zinb \
--condition_keys Cell_culture_batch \
--time_steps 1 3 \
--var_list Cell_population Cell_type Time_point Donor
echo "--- Finished computing model"
