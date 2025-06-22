#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=4 ' # request for exclusive access to gpu (:gmodel=NVIDIAA100_SXM4_80GB if you want to specify the gpu model)
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister # working directory
#BSUB -o logs/count_GF_frozen_imputation_%J.out # output file
#BSUB -e logs/count_GF_frozen_imputation_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_count_GF_frozen_imputation # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo "--- Start computing model"

# ----------------- Create folder to save results and copy the script -----------------
# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/iclr"
RES_NAME="cytoimmgen/imputation"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_all_timepoints_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_all_timepoints_$TIMESTAMP.sh"

# ----------------- Interpolation -----------------
# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/train.py \
--train_mode count \
--split True \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME/res \
--ckpt_masking_path "./T_perturb/T_perturb/iclr/cytoimmgen/imputation/res/checkpoints/20241001_0000_cellgen_train_masking_lr_1e-05_wd_1e-05_batch_64_psin_learnt_m_cosine_tp_1-2-3_s_42-epoch=09.ckpt" \
--src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_subsetted_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_subsetted_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl" \
--batch_size 64 \
--max_len 300 \
--epochs 5 \
--tgt_vocab_size 1254 \
--cellgen_lr 0.0001 \
--count_lr 0.005 \
--cellgen_wd 0.0001 \
--count_wd 0.001  \
--mlm_prob 0.15 \
--n_workers 4 \
--d_ff 64 \
--num_layers 6 \
--loss_mode zinb \
--condition_keys Cell_culture_batch \
--time_steps 1 2 3 \
--var_list Cell_population Cell_type Time_point Donor \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/output2/checkpoints/20250620_1508_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=04.ckpt" \
--pos_encoding_mode 'time_pos_sin' \
--seed 42 \
--mask_scheduler 'cosine'
echo "--- Finished computing model"
