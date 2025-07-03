#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister # working directory
#BSUB -o logs/random_pairing_generate_extra_%J.out # output file
#BSUB -e logs/random_pairing_generate_extra_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>150000] rusage[mem=150000]' # RAM memory part 1. Default: 100MB
#BSUB -J random_pairing_generate_extra # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister/iclr"
RES_NAME="cytoimmgen_extrapolation"
# if directory does not e
echo create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/4_run_val_generate_extrapolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/4_run_val_generate_extrapolation_$TIMESTAMP.sh"
# ----------------- Extrapolation -----------------
# python3 $cwd/val.py \
python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--generate True \
--ckpt_count_path './T_perturb/cytomeister/iclr/cytoimmgen_extrapolation/checkpoints/20240917_0836_cellgen_train_count_lr_0.005_wd_0.001_batch_256_zinb_tp_1-2_s_42-epoch=19.ckpt' \
--output_dir $RES_DIR/$RES_NAME \
--src_dataset "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_src_random_pairing_4096/0h.dataset" \
--tgt_dataset_folder "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_tgt_random_pairing_4096" \
--src_adata "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_src_random_pairing_4096/0h.h5ad" \
--tgt_adata_folder "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_tgt_random_pairing_4096" \
--mapping_dict_path  "./T_perturb/tokenized_data/cytoimmgen/token_id_to_genename_hvg.pkl" \
--batch_size 256 \
--max_len 291 \
--tgt_vocab_size 1254 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--num_layers 6 \
--loss_mode zinb \
--n_workers 32 \
--condition_keys Cell_culture_batch \
--time_steps 3 \
--var_list Cell_population Cell_type Time_point Donor \
--mode GF_fine_tuned \
--context_mode True
echo '--- Finished computing model'
