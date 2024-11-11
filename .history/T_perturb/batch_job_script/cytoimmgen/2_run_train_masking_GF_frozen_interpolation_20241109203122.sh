#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=shared:num=3" # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G team298 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb 
#BSUB -o T_perturb/log/interpolation_6h_out_opt_hparam%J.out # output file
#BSUB -e T_perturb/log/interpolation_6h_out_opt_hparam%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J interpolation # job name

# load cuda
# module load cuda-12.1.1

# activate pyenv
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/nature"
RES_NAME="LPS/interpolation_6h_out_opt_hparam"

# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh"

# ----------------- Interpolation -----------------
python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/train.py \
--train_mode masking \
--split False \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_10k/dataset_hvg_subsetted_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_10k/dataset_hvg_subsetted_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_10k/h5ad_pairing_hvg_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_10k/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/token_id_to_genename_hvg.pkl" \
--batch_size 64 \
--max_len 647 \
--epochs 50 \
--tgt_vocab_size 20274 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--mlm_prob 0.3 \
--n_workers 32 \
--d_ff 32 \
--num_layers 6 \
--time_steps 1 3 \
--var_list cell_type_cellgen_harm donor_cellgen_harm study time_after_LPS \
--mode GF_frozen \
--positional_encoding 'sin_learnt' \
--context_mode True \
--seed 42 \
--mask_scheduler 'cosine'
echo "--- Finished computing model"
