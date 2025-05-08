#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2:block=yes' # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB
#BSUB -n 16 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/cytoimmgen_masking_%J.out # output file
#BSUB -e logs/cytoimmgen_masking_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_masking # job name

# load cuda
module load cuda-12.1.1

# activate pyenvπ
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/plt/res"
RES_NAME="cytoimmgen/embedding"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_all_timepoints_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_all_timepoints_$TIMESTAMP.sh"

# ----------------- all_timepoints -----------------
# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/train.py \
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
--cellgen_lr 0.00001 \
--cellgen_wd 0.00001 \
--mlm_prob 0.15 \
--n_workers 16 \
--d_ff 64 \
--num_layers 6 \
--condition_keys Cell_culture_batch \
--pred_tps 1 2 3 \
--var_list Cell_population Cell_type Time_point Donor \
--cond_list Time_point \
--mode GF_frozen \
--context_mode True \
--mask_scheduler 'cosine' \
--positional_encoding 'sin_learnt'
echo "--- Finished computing model"
