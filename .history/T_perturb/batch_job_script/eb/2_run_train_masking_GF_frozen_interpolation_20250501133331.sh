#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-normal # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=shared:num=2' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -R "span[ptile=4]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/logs/eb_interpolation_6h_out_opt_hparam%J.out # output file
#BSUB -e T_perturb/logs/eb_interpolation_6h_out_opt_hparam%J.err # error file
#BSUB -M 20000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>20000] rusage[mem=20000]' # RAM memory part 1. Default: 100MB
#BSUB -J interpolation_eb # job name

# load cuda
#module load cuda-12.1.1

# activate python environment
#source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results"
RES_NAME="eb/pbmc_median/interpolation_CFG_off_test"
# # if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh"
# export WANDB_DIR=$cwd/wandb
# Run python script to PETRA
echo '--- Start computing model'

# # interpolation
# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/train.py \
--train_mode masking \
--split False \
--splitting_mode random \
--output_dir $RES_DIR/$RES_NAME \
--src_dataset '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/dataset_2000_hvg_src/Day 00-03.dataset' \
--tgt_dataset_folder '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/dataset_2000_hvg_tgt' \
--src_adata '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/h5ad_pairing_2000_hvg_src/Day 00-03.h5ad' \
--tgt_adata_folder '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/h5ad_pairing_2000_hvg_tgt' \
--mapping_dict_path  '/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/eb_pbmc_median/token_id_to_genename_2000_hvg.pkl' \
--batch_size 64 \
--max_len 300 \
--epochs 50 \
--tgt_vocab_size 1750 \
--cellgen_lr 0.001 \
--cellgen_wd 0.0001 \
--mlm_prob 0.15 \
--n_workers 4 \
--num_layers 3 \
--d_ff 32 \
--pred_tps  1 2 4 \
--var_list Time_point \
--cond_list Time_point \
--encoder scmaskgit \
--context_mode True \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--seed 100 \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--classifier_free_guidance False \
--d_model 768 \
> T_perturb/logs/eb_interpolation_$(date +%Y%m%d_%H%M%S).out \
2> T_perturb/logs/eb_interpolation_$(date +%Y%m%d_%H%M%S).err

echo '--- Finished computing model'

