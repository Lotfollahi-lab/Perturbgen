#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1:block=yes' # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/ipf_copd_embed_%J.out # output file
#BSUB -e logs/ipf_copd_embed_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'span[ptile=16]'  # Allocate 4 CPU cores per node
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J ipf_copd_embed # job name


# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # Run python script for rna
# python3 $cwd/val.py \
python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
--test_mode masking \
--split True \
--splitting_mode random \
--return_embed True \
--generate True \
--ckpt_masking_path "./T_perturb/T_perturb/Model/checkpoints/20240725_1726_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_32_mlmp_0.15_ntask_2_s_42-epoch=49.ckpt" \
--output_dir "./CellGen-reproducibility/ipf_copd/res" \
--src_dataset "./CellGen-reproducibility/ipf_copd/processed_data/dataset_hvg_src/Control.dataset" \
--tgt_dataset_folder "./CellGen-reproducibility/ipf_copd/processed_data/dataset_hvg_tgt/" \
--src_adata "./CellGen-reproducibility/ipf_copd/processed_data/h5ad_pairing_hvg_src/Control.h5ad" \
--tgt_adata_folder "./CellGen-reproducibility/ipf_copd/processed_data/h5ad_pairing_hvg_tgt" \
--cell_pairing_dir "./CellGen-reproducibility/ipf_copd/processed_data/cell_pairing" \
--tokenid_to_genename_dict  "./CellGen-reproducibility/ipf_copd/tokenid_to_genename.pkl" \
--batch_size 32 \
--max_len 1650 \
--tgt_vocab_size 25426 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--n_workers 32 \
--d_ff 128 \
--num_layers 1 \
--n_task_conditions 2 \
--var_list CellType_Category Manuscript_Identity Subclass_Cell_Identity Celltype_HLCA disease IPF_signature IPF_signature_disease profibrotic_mac_signature \
--encoder_type GF_frozen \
--moe_type none
echo "--- Finished computing model"
