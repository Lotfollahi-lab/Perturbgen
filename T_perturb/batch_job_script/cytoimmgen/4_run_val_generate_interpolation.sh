#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu :gmodel=NVIDIAH10080GBHBM3
#BSUB -n 32 # number of cores
#BSUB -R "span[ptile=32]"     # split X cores per host
#BSUB -G team298 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb 
#BSUB -o T_perturb/log/generate_inter_opthp_%J.out # output file
#BSUB -e T_perturb/log/generate_inter_opthp_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>150000] rusage[mem=150000]' # RAM memory part 1. Default: 100MB
#BSUB -J LPS_generate_inter # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/nature"
RES_NAME="LPS/generate_6h_interpolation_opthp_10k_ds_NP_zinb_s3000_ws_on3k_ds_tuned_3600"
# if directory does not e
# echo create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/4_run_val_generate_interpolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/4_run_val_generate_interpolation_$TIMESTAMP.sh"
# ----------------- Extrapolation -----------------
# python3 $cwd/val.py \
python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--generate True \
--ckpt_count_path '/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/nature/LPS/count_decoder_6h_interpolation_opthp_10k_final_model_ws_on3k_ds_tuned_3600/res/checkpoints/20241230_1508_cellgen_train_count_lr_0.001_wd_0.0001_batch_64_zinb_tp_1-3_s_42_pos_sin_learnt_m_cosine-epoch=02.ckpt' \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_3k_ds/dataset_hvg_subsetted_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_3k_ds/dataset_hvg_subsetted_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_3k_ds/h5ad_pairing_hvg_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_3k_ds/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/token_id_to_genename_hvg.pkl" \
--batch_size 16 \
--max_len 3600 \
--tgt_vocab_size 20274 \
--count_lr 0.001 \
--cellgen_lr 0.00005 \
--cellgen_wd 0.00001 \
--count_wd 0.0001  \
--sequence_length 3000 \
--d_ff 32 \
--num_layers 6 \
--loss_mode zinb \
--n_samples 3 \
--n_workers 32 \
--time_steps 2 \
--var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS \
--mode GF_fine_tuned \
--mask_scheduler 'cosine' \
--positional_encoding sin_learnt \
--seed 42 \
--context_mode True 
echo '--- Finished computing model'

