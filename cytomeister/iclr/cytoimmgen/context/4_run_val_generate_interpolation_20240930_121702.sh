#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu (:gmodel=NVIDIAA100_SXM4_80GB if you want to specify the gpu model)
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister # working directory
#BSUB -o logs/cyto_generate_context_%J.out # output file
#BSUB -e logs/cyto_generate_context_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_generate_context # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister/iclr"
RES_NAME="cytoimmgen/context"
# if directory does not e
echo create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/4_run_val_generate_interpolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/4_run_val_generate_interpolation_$TIMESTAMP.sh"

# ----------------- Interpolation -----------------
# python3 $cwd/val.py \
python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--generate True \
--ckpt_count_path './T_perturb/cytomeister/iclr/cytoimmgen/context/res/checkpoints/20240930_1148_cellgen_train_count_lr_0.005_wd_0.001_batch_64_zinb_tp_1-3_s_42_pos_sin_learnt_m_cosine-epoch=01.ckpt' \
--output_dir $RES_DIR/$RES_NAME \
--src_dataset "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_subsetted_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_subsetted_tgt" \
--src_adata "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "./T_perturb/Geneformer/geneformer/token_dictionary_gc95M.pkl" \
--batch_size 128 \
--max_len 291 \
--tgt_vocab_size 1254 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.005 \
--count_wd 0.001 \
--num_layers 6 \
--d_ff 64 \
--loss_mode zinb \
--n_workers 32 \
--condition_keys Cell_culture_batch \
--time_steps 2 \
--var_list Cell_population Cell_type Time_point Donor \
--mode GF_frozen \
--seed 42 \
--positional_encoding sin_learnt \
--context_mode False \
--mask_scheduler 'cosine'
echo '--- Finished computing model'
