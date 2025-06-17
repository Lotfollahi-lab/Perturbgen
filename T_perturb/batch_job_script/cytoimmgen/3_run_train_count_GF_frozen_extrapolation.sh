#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=3:block=yes' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/count_extra_s100_%J.out # output file
#BSUB -e logs/count_extra_s100_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_count_extra_s0 # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
RES_DIR="/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/plt/res"
RES_NAME="cytoimmgen/pbmc_median/extrapolation"
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo "--- Start computing model"

# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
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
--output_dir $RES_DIR/$RES_NAME/res \
--ckpt_masking_path "T_perturb/T_perturb/plt/res/cytoimmgen/pbmc_median/extrapolation/res/checkpoints/20250512_2125_cellgen_train_masking_lr_1e-05_wd_1e-05_batch_64_ptime_pos_sin_m_pow_tp_1-2_s_100-epoch=19.ckpt" \
--src_dataset "T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/dataset_2000_hvg_src/0h.dataset" \
--tgt_dataset_folder "T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/dataset_2000_hvg_tgt" \
--src_adata "T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/h5ad_pairing_2000_hvg_src/0h.h5ad" \
--tgt_adata_folder "T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path  "T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 400 \
--epochs 4 \
--tgt_vocab_size 1360 \
--count_lr 0.001 \
--count_wd 0.001  \
--count_dropout 0.0 \
--n_workers 4 \
--d_ff 64 \
--num_layers 6 \
--loss_mode zinb \
--d_model 768 \
--condition_keys Cell_culture_batch \
--pred_tps 1 2 \
--cond_list Time_point \
--var_list Cell_population Cell_type Time_point Donor \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--add_cell_time False \
--use_positional_encoding False \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--seed 100 \
--context_mode True \
--use_weighted_sampler False
echo "--- Finished computing model"
