#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=shared:num=4' # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -R "span[ptile=16]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/logs/interpolation_6h_out_opt_hparam%J.out # output file
#BSUB -e T_perturb/logs/interpolation_6h_out_opt_hparam%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J interpolation_2k # job name

# activate pyenv
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)
## source /lustre/scratch126/cellgen/team361/av13/scmaskgit/.venv/bin/activate

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results"
RES_NAME="lps/interpolation_2k_noControl_oldPairing"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP_$SLURM_JOB_ID.sh
echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh"

# export WANDB_DIR=$cwd/wandb
# Run python script to PETRA
echo '--- Start computing model'

# # interpolation

python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/train.py \
--train_mode masking \
--split False \
--splitting_mode stratified \
--split_obs cell_type_cellgen_harm \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps_butNormal_oldPairing/dataset_2000_hvg_src/90m_LPS.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps_butNormal_oldPairing/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps_butNormal_oldPairing/h5ad_pairing_2000_hvg_src/90m_LPS.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps_butNormal_oldPairing/dataset_2000_hvg_tgt" \
--mapping_dict_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps_butNormal_oldPairing/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 725 \
--epochs 26 \
--tgt_vocab_size 20274 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--mlm_prob 0.3 \
--n_workers 16 \
--num_layers 6 \
--d_ff 32 \
--pred_tps 1 2 \
--var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--seed 42 \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'cosine' \
--num_node 1 \
--d_model 768

echo '--- Finished computing model'

