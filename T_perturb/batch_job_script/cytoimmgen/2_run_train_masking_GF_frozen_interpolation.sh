#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=shared:num=2' # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -R "span[ptile=16]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/logs/tcell_interpolation_6h_out_opt_hparam%J.out # output file
#BSUB -e T_perturb/logs/tcell_interpolation_6h_out_opt_hparam%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J interpolation_tcell # job name

# load cuda
#module load cuda-12.1.1

# activate pyenv
#source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/results"
RES_NAME="cytoimmgen/pbmc_median/interpolation"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh"

# ----------------- Interpolation -----------------
python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/train.py \
--train_mode masking \
--split False \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME \
--src_dataset "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/dataset_2000_hvg_src/0h.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/h5ad_pairing_2000_hvg_src/0h.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path  "/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/pp/res/cytoimmgen_pbmc_median/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 400 \
--epochs 10 \
--tgt_vocab_size 1360 \
--cellgen_lr 0.00001 \
--cellgen_wd 0.00001 \
--n_workers 4 \
--d_ff 64 \
--num_layers 6 \
--condition_keys Cell_culture_batch \
--pred_tps 1 3 \
--var_list Cell_population Cell_type Time_point Donor \
--encoder scmaskgit \
--context_mode True \
--cond_list Time_point \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--pos_encoding_mode time_pos_sin \
--seed 100 \
--mask_scheduler 'pow' \
--d_model 768
echo "--- Finished computing model"
