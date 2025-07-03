#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=4:block=yes' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister # working directory
#BSUB -o logs/count_GF_frozen_extra_%J.out # output file
#BSUB -e logs/count_GF_frozen_extra_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_count_GF_frozen_extra # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo "--- Start computing model"

# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister/iclr"
RES_NAME="cytoimmgen_extrapolation"
# # if directory does not exist, create it with the name $RES_NAME
# mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/3_run_train_count_GF_frozen_extrapolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/3_run_train_count_GF_frozen_extrapolation_$TIMESTAMP.sh"

# ----------------- Extrapolation -----------------
python3 $cwd/train.py \
--train_mode count \
--split False \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME \
--ckpt_masking_path "./T_perturb/cytomeister/iclr/cytoimmgen_extrapolation/checkpoints/20240917_0722_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_256_mlmp_0.15_tp_1-2_s_42-epoch=49.ckpt" \
--src_dataset "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_src_random_pairing_4096/0h.dataset" \
--tgt_dataset_folder "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_tgt_random_pairing_4096" \
--src_adata "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_src_random_pairing_4096/0h.h5ad" \
--tgt_adata_folder "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_tgt_random_pairing_4096" \
--mapping_dict_path  "./T_perturb/tokenized_data/cytoimmgen/token_id_to_genename_hvg.pkl" \
--batch_size 256 \
--max_len 291 \
--epochs 20 \
--tgt_vocab_size 1254 \
--cellgen_lr 0.0001 \
--count_lr 0.005 \
--cellgen_wd 0.0001 \
--count_wd 0.001  \
--mlm_prob 0.15 \
--n_workers 32 \
--d_ff 128 \
--num_layers 6 \
--loss_mode zinb \
--condition_keys Cell_culture_batch \
--time_steps 1 2 \
--var_list Cell_population Cell_type Time_point Donor \
--mode GF_frozen \
--context_mode True
echo "--- Finished computing model"
