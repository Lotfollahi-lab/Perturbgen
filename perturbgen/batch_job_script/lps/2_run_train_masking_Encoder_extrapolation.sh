#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=4:gmodel=NVIDIAA100_SXM4_80GB' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -R "span[ptile=4]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister # working directory
#BSUB -o logs/lps_extrapolation_2k_%J.out # output file
#BSUB -e logs/lps_extrapolation_2k_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J lps_extrapolation_2k # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/"
RES_NAME="lps/extrapolation/"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME

# # extrapolation
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen/train.py \
--train_mode masking \
--split False \
--splitting_mode stratified \
--split_obs cell_type_harmonized \
--output_dir $RES_DIR/$RES_NAME \
--src_dataset "T_perturb/tokenized_data/lps_100M/dataset_2000_hvg_src/normal.dataset" \
--tgt_dataset_folder "T_perturb/tokenized_data/lps_100M/dataset_2000_hvg_tgt" \
--src_adata "T_perturb/tokenized_data/lps_100M/h5ad_pairing_2000_hvg_src/normal.h5ad" \
--tgt_adata_folder "T_perturb/tokenized_data/lps_100M/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "T_perturb/tokenized_data/lps_100M/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--epochs 20 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--n_workers 4 \
--num_layers 6 \
--d_ff 64 \
--pred_tps 1 2 \
--var_list cell_type_harmonized cell_pairing_index time_after_LPS \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt" \
--seed 100 \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--num_node 1 \
--d_model 768 \
--use_weighted_sampler True \
--sampling_keys cell_type_harmonized \
--ckpt_every_n_epochs 5

echo '--- Finished computing model'
