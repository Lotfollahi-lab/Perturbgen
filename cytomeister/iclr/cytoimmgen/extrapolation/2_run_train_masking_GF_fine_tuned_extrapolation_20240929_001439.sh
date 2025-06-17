#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=4:block=yes' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/cytoimmgen_masking_extra_s0_%J.out # output file
#BSUB -e logs/cytoimmgen_masking_extra_s0_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_masking_extra_s100 # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/iclr"
RES_NAME="cytoimmgen/extrapolation"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_fine_tuned_extrapolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_fine_tuned_extrapolation_$TIMESTAMP.sh"

# ----------------- Extrapolation -----------------
python3 $cwd/train.py \
--train_mode masking \
--split False \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_subsetted_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_subsetted_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl" \
--batch_size 64 \
--max_len 300 \
--epochs 10 \
--tgt_vocab_size 1254 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--mlm_prob 0.15 \
--n_workers 16 \
--d_ff 64 \
--num_layers 6 \
--condition_keys Cell_culture_batch \
--time_steps 1 2 \
--var_list Cell_population Cell_type Time_point Donor \
--mode GF_frozen \
--context_mode True \
--positional_encoding 'sin_learnt' \
--seed 0 \
--mask_scheduler 'cosine'
echo "--- Finished computing model"
