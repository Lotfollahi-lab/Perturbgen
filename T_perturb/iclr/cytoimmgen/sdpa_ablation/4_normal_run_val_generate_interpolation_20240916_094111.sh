#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/sdpa_generate_inter_%J.out # output file
#BSUB -e logs/sdpa_generate_inter=_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>150000] rusage[mem=150000]' # RAM memory part 1. Default: 100MB
#BSUB -J sdpa_generate_inter # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/iclr"
RES_NAME="sdpa_ablation"
# if directory does not e
echo create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/4_sdpa_run_val_generate_interpolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/4_sdpa_val_generate_interpolation_$TIMESTAMP.sh"

# ----------------- Interpolation -----------------
# python3 $cwd/val.py \
python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--generate True \
--ckpt_count_path './T_perturb/T_perturb/iclr/sdpa_ablation/checkpoints/20240916_0757_sdpa_train_count_lr_0.005_wd_0.001_batch_64_zinb_tp_1-3_s_42-epoch=09.ckpt' \
--output_dir './T_perturb/T_perturb/plt/res/cytoimmgen' \
--src_dataset './T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset' \
--tgt_dataset_folder './T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt' \
--src_adata './T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad' \
--tgt_adata_folder './T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt' \
--batch_size 256 \
--max_len 300 \
--tgt_vocab_size 1261 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--num_layers 6 \
--loss_mode zinb \
--n_workers 32 \
--condition_keys Cell_culture_batch \
--time_steps 2 \
--var_list Cell_population Cell_type Time_point Donor \
--mode GF_fine_tuned
echo '--- Finished computing model'
