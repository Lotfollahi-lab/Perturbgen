#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1:gmodel=NVIDIAA100_SXM4_80GB' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister # working directory
#BSUB -o logs/cyto_generate_temra_%J.out # output file
#BSUB -e logs/cyto_generate_temra_%J.err # error file
#BSUB -M 100000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>100000] rusage[mem=100000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_generate_temra # job name

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
RES_NAME="cytoimmgen/generation"
# if directory does not e
echo create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/4_run_val_generate_temrapolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/4_run_val_generate_temrapolation_$TIMESTAMP.sh"

# ----------------- Interpolation -----------------
# python3 $cwd/val.py \
python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--generate True \
--ckpt_count_path './T_perturb/cytomeister/iclr/cytoimmgen/masking_scheduler/res/checkpoints/20240921_1828_cellgen_train_count_lr_0.005_wd_0.001_batch_64_zinb_tp_1-3_s_42_mask_exp-epoch=01.ckpt' \
--output_dir $RES_DIR/$RES_NAME \
--src_dataset "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_tgt" \
--src_adata "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "./T_perturb/Geneformer/geneformer/token_dictionary_gc95M.pkl" \
--batch_size 256 \
--max_len 291 \
--tgt_vocab_size 20274 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--num_layers 6 \
--loss_mode zinb \
--n_workers 32 \
--condition_keys Cell_culture_batch \
--time_steps 1 \
--var_list Cell_population Cell_type Time_point Donor \
--mode GF_frozen \
--mask_scheduler 'exp' \
--guided_gene_list_dir None \
--hvg_gene_list_dir "./T_perturb/cytomeister/iclr/cytoimmgen/generation/hvg_list.pkl" \
--generate_cell_type 'TEMRA 16h'
echo '--- Finished computing model'
