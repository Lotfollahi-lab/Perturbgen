#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2:block=yes' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -R "span[ptile=4]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb # working directory
#BSUB -o T_perturb/log/lps_count_interpolation_2k_%J.out # output file
#BSUB -e T_perturb/log/lps_count_interpolation_2k_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J lps_count_interpolation_2k # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
# source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
# results directory
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/plt/res"
RES_NAME="lps/ourMED_median/count_noNormal_Med100Me01_cosine_e8_lps"
# # if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # interpolation
python3 /lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/train.py \
--train_mode count \
--split False \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME \
--ckpt_masking_path "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/plt/res/lps/Our_median/interpolation_LPS_e1_100M_CondTime_cosine_withControl_2k_noNormal/res/checkpoints/20250720_1247_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_cosine_tp_1-2_s_0-epoch=08.ckpt" \
--src_dataset "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg_noNormal/dataset_2000_hvg_src/90m_LPS.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg_noNormal/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg_noNormal/h5ad_pairing_2000_hvg_src/90m_LPS.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg_noNormal/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 797 \
--epochs 8 \
--tgt_vocab_size 2002 \
--count_lr 0.001 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_wd 0.001 \
--count_dropout 0.1 \
--n_workers 4 \
--num_layers 6 \
--d_ff 32 \
--loss_mode zinb \
--pred_tps 1 2 \
--var_list cell_pairing_index time_after_LPS cell_type_harmonized \
--cond_list time_after_LPS \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=01.ckpt" \
--add_cell_time False \
--use_positional_encoding False \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'cosine' \
--num_node 1 \
--d_model 768 \
--seed 42 \
--use_weighted_sampler False

echo '--- Finished computing model'

# --encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
