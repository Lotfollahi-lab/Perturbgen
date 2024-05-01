#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu
#BSUB -n 64 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/masking_%J.out # output file
#BSUB -e logs/masking_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000] span[ptile=1]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_masking # job name
#BSUB -W 12:00 # time for the job HH:MM:SS. Default: 1 min


# activate conda environment
source ~/.bashrc
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"
# # Run python script for rna
python3 $cwd/train.py \
--train_mode masking \
--split True \
--splitting_mode stratified \
--generate False \
--num_cells 0 \
--src_dataset "./T_perturb/T_perturb/pp/cytoimmgen/dataset_hvg_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/cytoimmgen/dataset_hvg_tgt" \
--src_adata "./T_perturb/T_perturb/pp/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/cytoimmgen/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "./T_perturb/T_perturb/pp/cytoimmgen/token_id_to_genename_hvg.pkl" \
--batch_size 64 \
--max_len 400 \
--epochs 100 \
--tgt_voc_size 1820 \
--petra_lr 0.001 \
--count_lr 0.0005 \
--petra_wd 0.001 \
--count_wd 0.001 \
--mlm_prob 0.15 \
--n_workers 64 \
--condition_keys Cell_culture_batch \
--time_steps 1 2 3 \
--var_list Cell_population Cell_type Time_point Donor \
echo "--- Finished computing model"
