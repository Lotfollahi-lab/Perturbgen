#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=6:block=yes' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/count_%J.out # output file
#BSUB -e logs/count_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_count # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo "--- Start computing model"

#interpolation
python3 $cwd/train.py \
--train_mode count \
--split False \
--splitting_mode stratified \
--output_dir "./T_perturb/T_perturb/plt/res/cytoimmgen" \
--ckpt_masking_path "./T_perturb/T_perturb/Model/checkpoints/20240519_0000_GF_finetuned"\
"_lr_0.0001_wd_0.0001_batch_64_mlmp_0.15_tp_1-3-epoch=49.ckpt" \
--src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl" \
--batch_size 64 \
--max_len 300 \
--epochs 20 \
--tgt_vocab_size 1261 \
--petra_lr 0.0001 \
--count_lr 0.005 \
--petra_wd 0.0001 \
--count_wd 0.001  \
--mlm_prob 0.15 \
--n_workers 32 \
--d_ff 128 \
--num_layers 6 \
--loss_mode zinb \
--condition_keys Cell_culture_batch \
--time_steps 1 3 \
--var_list Cell_population Cell_type Time_point Donor
--mode GF_fine_tuned
echo "--- Finished computing model"

# # extrapolation
# python3 $cwd/train.py \
# --train_mode count \
# --split False \
# --splitting_mode stratified \
# --output_dir "./T_perturb/T_perturb/plt/res/cytoimmgen" \
# --ckpt_masking_path "./T_perturb/T_perturb/Model/checkpoints/20240519_1114_Tcell"\
# "_extrapol_GF_fine_tuned_lr_0.0001_wd_0.0001_batch_64_mlmp_0.15_tp_1-2-epoch=49.ckpt" \
# --src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset" \
# --tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt" \
# --src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
# --tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
# --mapping_dict_path  "./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl" \
# --batch_size 64 \
# --max_len 300 \
# --epochs 20 \
# --tgt_vocab_size 1261 \
# --petra_lr 0.0001 \
# --count_lr 0.005 \
# --petra_wd 0.0001 \
# --count_wd 0.001 \
# --mlm_prob 0.15 \
# --n_workers 32 \
# --d_ff 128 \
# --num_layers 6 \
# --loss_mode zinb \
# --condition_keys Cell_culture_batch \
# --time_steps 1 2 \
# --var_list Cell_population Cell_type Time_point Donor \
# --mode GF_fine_tuned
# echo "--- Finished computing model"
