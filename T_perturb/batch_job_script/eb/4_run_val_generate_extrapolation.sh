#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu, # request for exclusive access to gpu, 'mode=shared:num=2:gmem=10000'
#BSUB -n 4 # number of cores
#BSUB -R "span[ptile=4]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/logs/eb_generate_extra_%J.out # output file
#BSUB -e T_perturb/logs/eb_generate_extra_%J.err # error file
#BSUB -M 20000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>20000] rusage[mem=20000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_generate_extra # job name

# load cuda
#module load cuda-12.1.1

# activate pyenv
#source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# # run script
echo '--- Start computing model'

RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results"
RES_NAME="eb/pbmc_median/extrapolation"

# # if directory does not exist, create it with the name $RES_NAME
# mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/4_run_val_generate_extrapolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/4_run_val_generate_extrapolation_$TIMESTAMP.sh"

# extrapolation
python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/val.py \
--test_mode count \
--split False \
--splitting_mode random \
--generate True \
--output_dir $RES_DIR/$RES_NAME/res \
--ckpt_count_path '/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results/eb/pbmc_median/extrapolation/checkpoints/20250426_0058_cellgen_train_count_lr_0.0001_wd_0.0001_batch_64_zinb_tp_1-2-3_s_0_pos_time_pos_sin_m_pow-epoch=99.ckpt' \
--src_dataset '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/dataset_2000_hvg_src/Day 00-03.dataset' \
--tgt_dataset_folder '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/dataset_2000_hvg_tgt' \
--src_adata '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/h5ad_pairing_2000_hvg_src/Day 00-03.h5ad' \
--tgt_adata_folder '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/h5ad_pairing_2000_hvg_tgt' \
--mapping_dict_path  '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/token_id_to_genename_2000_hvg.pkl' \
--batch_size 64 \
--max_len 300 \
--tgt_vocab_size 1750 \
--cellgen_lr 0.001 \
--cellgen_wd 0.0001 \
--count_lr 0.0001 \
--count_wd 0.0001 \
--num_layers 3 \
--d_ff 32 \
--loss_mode zinb \
--n_workers 4 \
--pred_tps 4 \
--context_tps 1 2 3 \
--var_list Time_point \
--cond_list Time_point \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--temperature 1.0 \
--sequence_length 150 \
--iterations 20 \
--n_samples 3 \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--d_model 768 \
--seed 0
echo '--- Finished computing model'
