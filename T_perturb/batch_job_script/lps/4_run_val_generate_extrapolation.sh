#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -R "span[ptile=4]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/lps_generate_extra_s42_%J.out # output file
#BSUB -e logs/lps_generate_extra_s42_%J.err # error file
#BSUB -M 100000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>100000] rusage[mem=100000]' # RAM memory part 1. Default: 100MB
#BSUB -J lps_generate_extra_s42 # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'

RES_DIR="/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/plt/res"
RES_NAME="lps/pbmc_median/extrapolation"



python3 $cwd/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--generate True \
--output_dir $RES_DIR/$RES_NAME/res \
--ckpt_count_path 'T_perturb/T_perturb/plt/res/lps/pbmc_median/extrapolation/res/checkpoints/20250513_1751_cellgen_train_count_lr_0.001_wd_0.0001_batch_64_zinb_tp_1-2_s_42_pos_time_pos_sin_m_pow-epoch=04.ckpt' \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/dataset_2000_hvg_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/res/2k_hvg_ourMED_all_tps/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 666 \
--tgt_vocab_size 1990 \
--count_lr 0.001 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--sequence_length 150 \
--count_wd 0.001 \
--num_layers 6 \
--d_ff 64 \
--loss_mode zinb \
--n_workers 4 \
--pred_tps 3 \
--context_tps 1 2 \
--var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS \
--cond_list time_after_LPS \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--add_cell_time False \
--use_positional_encoding False \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--iterations 20 \
--temperature 0.5 \
--n_samples 2 \
--num_node 1 \
--d_model 768 \
--seed 42
echo '--- Finished computing model'
