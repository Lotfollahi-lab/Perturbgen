#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi-train # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=4' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -R "span[ptile=4]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen # working directory
#BSUB -o logs/lps_count_extrapolation_s0_ep09_%J.out # output file
#BSUB -e logs/lps_count_extrapolation_s0_ep09_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J lps_count_extrapolation_s0_ep09 # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
# results directory
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/"
RES_NAME="lps/extrapolation"
# # if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # extrapolation
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen/train.py \
--train_mode count \
--split False \
--splitting_mode stratified \
--split_obs cell_type_harmonized \
--output_dir $RES_DIR/$RES_NAME \
--ckpt_masking_path "T_perturb/res/lps/extrapolation/checkpoints/20250829_1348_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1-2_s_0-epoch=09.ckpt" \
--src_dataset "T_perturb/tokenized_data/lps_100M/dataset_2000_hvg_src/normal.dataset" \
--tgt_dataset_folder "T_perturb/tokenized_data/lps_100M/dataset_2000_hvg_tgt" \
--src_adata "T_perturb/tokenized_data/lps_100M/h5ad_pairing_2000_hvg_src/normal.h5ad" \
--tgt_adata_folder "T_perturb/tokenized_data/lps_100M/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "T_perturb/tokenized_data/lps_100M/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--epochs 5 \
--count_lr 0.001 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_wd 0.0001 \
--count_dropout 0.1 \
--n_workers 4 \
--num_layers 6 \
--d_ff 64 \
--loss_mode zinb \
--pred_tps 1 2 \
--var_list cell_type_harmonized cell_pairing_index time_after_LPS \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt" \
--add_cell_time False \
--use_positional_encoding False \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--num_node 1 \
--d_model 768 \
--seed 0 \
--use_weighted_sampler True \
--sampling_keys cell_type_harmonized \
--ckpt_every_n_epochs 1

echo '--- Finished computing model'
