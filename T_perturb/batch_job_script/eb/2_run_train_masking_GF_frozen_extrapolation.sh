#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/eb_masking_extra_s0_%J.out # output file
#BSUB -e logs/eb_masking_extra_s0_%J.err # error file
#BSUB -M 20000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>20000] rusage[mem=20000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_masking_extra_s0 # job name

# load cuda
module load cuda-12.1.1

# activate python environment
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/plt/res"
RES_NAME="eb/pbmc_median/extrapolation"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_extrapolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_extrapolation_$TIMESTAMP.sh"

# export WANDB_DIR=$cwd/wandb
# Run python script to PETRA
echo '--- Start computing model'

# ----------------- Extrapolation -----------------
# # python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/train.py \
--train_mode masking \
--split False \
--splitting_mode random \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset 'T_perturb/T_perturb/pp/res/eb_pbmc_median/dataset_2000_hvg_src/Day 00-03.dataset' \
--tgt_dataset_folder 'T_perturb/T_perturb/pp/res/eb_pbmc_median/dataset_2000_hvg_tgt' \
--src_adata 'T_perturb/T_perturb/pp/res/eb_pbmc_median/h5ad_pairing_2000_hvg_src/Day 00-03.h5ad' \
--tgt_adata_folder 'T_perturb/T_perturb/pp/res/eb_pbmc_median/h5ad_pairing_2000_hvg_tgt' \
--mapping_dict_path  'T_perturb/T_perturb/pp/res/eb_pbmc_median/token_id_to_genename_2000_hvg.pkl' \
--batch_size 64 \
--max_len 300 \
--epochs 50 \
--tgt_vocab_size 1750 \
--cellgen_lr 0.001 \
--cellgen_wd 0.0001 \
--n_workers 4 \
--num_layers 3 \
--d_ff 32 \
--pred_tps 1 2 3 \
--var_list Time_point \
--cond_list Time_point \
--encoder scmaskgit \
--context_mode True \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--seed 0 \
--d_model 768 \
--use_weighted_sampler False
echo '--- Finished computing model'

# PBMC median
# --encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \

# 20k GF median
# --encoder_path '/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output2/checkpoints/20250110_2325_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt' \
# --tgt_vocab_size 1730 \
# --max_len 270 \
